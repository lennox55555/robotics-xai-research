#!/usr/bin/env python3
"""
Web-based MuJoCo Viewer

Serves the G1 robot simulation on localhost for viewing in a browser.

Usage:
    python web_viewer.py --skill walk_forward
    # Then open http://localhost:8000 in your browser
"""

import argparse
import sys
import time
import base64
import io
import threading
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco
from PIL import Image
from flask import Flask, Response, render_template_string

# Global state
app = Flask(__name__)
simulation_state = {
    "model": None,
    "data": None,
    "renderer": None,
    "policy": None,
    "running": True,
    "fps": 30,
    "skill_id": None,
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>G1 Robot Viewer</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .info {
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .info p {
            margin: 5px 0;
        }
        .skill-name {
            color: #00ff88;
            font-weight: bold;
        }
        .video-container {
            background: #0f0f23;
            border-radius: 8px;
            padding: 10px;
            display: inline-block;
        }
        img {
            border-radius: 4px;
        }
        .stats {
            margin-top: 20px;
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
        }
        .controls {
            margin-top: 20px;
        }
        button {
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        button:hover {
            background: #00a8cc;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>G1 Humanoid Robot Viewer</h1>

        <div class="info">
            <p>Skill: <span class="skill-name">{{ skill_id or "None (random actions)" }}</span></p>
            <p>Robot: Unitree G1 with Hands (44 DOF)</p>
            <p>Simulation: MuJoCo</p>
        </div>

        <div class="video-container">
            <img src="/video_feed" width="800" height="600">
        </div>

        <div class="stats" id="stats">
            <p>Streaming simulation...</p>
        </div>

        <div class="controls">
            <button onclick="location.reload()">Refresh</button>
            <a href="/reset"><button>Reset Simulation</button></a>
        </div>
    </div>

    <script>
        // Update stats periodically
        setInterval(async () => {
            try {
                const resp = await fetch('/stats');
                const data = await resp.json();
                document.getElementById('stats').innerHTML = `
                    <p>Step: ${data.step}</p>
                    <p>COM Height: ${data.com_height.toFixed(3)}m</p>
                    <p>Forward Velocity: ${data.forward_vel.toFixed(3)} m/s</p>
                    <p>Reward: ${data.reward.toFixed(3)}</p>
                `;
            } catch (e) {}
        }, 200);
    </script>
</body>
</html>
"""


def load_model():
    """Load the MuJoCo model."""
    xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
    simulation_state["model"] = mujoco.MjModel.from_xml_path(str(xml_path))
    simulation_state["data"] = mujoco.MjData(simulation_state["model"])
    simulation_state["renderer"] = mujoco.Renderer(
        simulation_state["model"],
        height=600,
        width=800
    )
    mujoco.mj_forward(simulation_state["model"], simulation_state["data"])


def load_policy(skill_id: str):
    """Load a trained policy."""
    if not skill_id:
        return None

    model_path = PROJECT_ROOT / "skills" / "trained" / skill_id / "model.zip"
    if model_path.exists():
        from stable_baselines3 import PPO
        print(f"Loading trained model: {skill_id}")
        simulation_state["policy"] = PPO.load(str(model_path))
        simulation_state["skill_id"] = skill_id
        return simulation_state["policy"]
    else:
        print(f"No trained model found for: {skill_id}")
        return None


def get_observation():
    """Get observation vector."""
    data = simulation_state["data"]
    model = simulation_state["model"]

    qpos = data.qpos[3:].copy()  # Skip root xyz
    qvel = data.qvel.copy()
    torso_xmat = data.xmat[1].reshape(9)

    return np.concatenate([qpos, qvel, torso_xmat])


def step_simulation():
    """Step the simulation forward."""
    model = simulation_state["model"]
    data = simulation_state["data"]
    policy = simulation_state["policy"]

    # Get action
    if policy is not None:
        obs = get_observation()
        action, _ = policy.predict(obs, deterministic=True)
        data.ctrl[:] = action * model.actuator_ctrlrange[:, 1]
    else:
        # Small random actions
        low = model.actuator_ctrlrange[:, 0]
        high = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.random.uniform(low, high) * 0.05

    # Step physics
    for _ in range(5):  # Frame skip
        mujoco.mj_step(model, data)


def render_frame():
    """Render current frame as JPEG bytes."""
    renderer = simulation_state["renderer"]
    data = simulation_state["data"]

    renderer.update_scene(data)
    pixels = renderer.render()

    # Convert to JPEG
    img = Image.fromarray(pixels)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


def generate_frames():
    """Generator that yields video frames."""
    step = 0
    while simulation_state["running"]:
        # Step simulation
        step_simulation()
        step += 1

        # Store stats
        data = simulation_state["data"]
        simulation_state["step"] = step
        simulation_state["com_height"] = float(data.subtree_com[0][2])
        simulation_state["forward_vel"] = float(data.qvel[0])

        # Compute simple reward
        upright = data.xmat[1].reshape(3, 3)[2, 2]
        simulation_state["reward"] = upright + simulation_state["com_height"]

        # Render frame
        frame = render_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Control framerate
        time.sleep(1.0 / simulation_state["fps"])


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE, skill_id=simulation_state["skill_id"])


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    """Return current simulation stats."""
    import json
    return json.dumps({
        "step": simulation_state.get("step", 0),
        "com_height": simulation_state.get("com_height", 0),
        "forward_vel": simulation_state.get("forward_vel", 0),
        "reward": simulation_state.get("reward", 0),
    })


@app.route('/reset')
def reset():
    """Reset the simulation."""
    mujoco.mj_resetData(simulation_state["model"], simulation_state["data"])
    mujoco.mj_forward(simulation_state["model"], simulation_state["data"])
    return '<script>window.location.href="/"</script>'


def main():
    parser = argparse.ArgumentParser(description="Web-based G1 robot viewer")
    parser.add_argument("--skill", type=str, help="Skill to load and execute")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")

    args = parser.parse_args()

    simulation_state["fps"] = args.fps

    print("\n" + "="*50)
    print("G1 Robot Web Viewer")
    print("="*50)

    # Load model
    print("Loading MuJoCo model...")
    load_model()

    # Load policy if specified
    if args.skill:
        load_policy(args.skill)

    print(f"\nStarting server on http://localhost:{args.port}")
    print("Open this URL in your browser to view the simulation")
    print("Press Ctrl+C to stop\n")

    # Run Flask
    app.run(host='0.0.0.0', port=args.port, threaded=True)


if __name__ == "__main__":
    main()
