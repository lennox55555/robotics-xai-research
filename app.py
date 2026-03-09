#!/usr/bin/env python3
"""
G1 Robot Skill Learning Web Application

A web-based interface for teaching the G1 humanoid robot new skills.
Combines:
- Chat interface with the AI orchestrator
- Real-time MuJoCo simulation viewer
- Training progress monitoring

Usage:
    python app.py
    # Open http://localhost:5000 in your browser
"""

import os
import sys
import json
import time
import io
import threading
import queue
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco
from PIL import Image
from flask import Flask, Response, render_template_string, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# Import our modules
from src.agents.orchestrator_v2 import UnifiedOrchestrator
from src.robot.robot_spec import SKILL_TEMPLATES, get_robot_spec
from src.experiments.experiment_runner import train_skill as run_training

app = Flask(__name__)

# Global state
state = {
    "orchestrator": None,
    "model": None,
    "data": None,
    "renderer": None,
    "policy": None,
    "vec_normalize": None,  # For observation normalization
    "current_skill": None,
    "running": True,
    "step": 0,
    "messages": [],
    "training_active": False,
    "training_progress": 0,
    "training_skill": None,
    "training_timesteps": 0,
    "training_total": 0,
    "latest_frame": None,
    "frame_lock": threading.Lock(),
}


def training_thread(skill_id: str, timesteps: int):
    """Background thread for training a skill."""
    # Clear any problematic MuJoCo GL settings for macOS
    os.environ.pop('MUJOCO_GL', None)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    state["training_active"] = True
    state["training_skill"] = skill_id
    state["training_total"] = timesteps
    state["training_timesteps"] = 0

    class LiveViewCallback(BaseCallback):
        """Callback to update training progress and reload policy for viewing."""
        def __init__(self):
            super().__init__()
            self.last_reload = 0

        def _on_step(self) -> bool:
            state["training_timesteps"] = self.num_timesteps

            # Every 5k steps, save and reload the policy for live viewing
            if self.num_timesteps - self.last_reload >= 5000:
                self.last_reload = self.num_timesteps
                # Save current model and normalization
                model_path = PROJECT_ROOT / "skills" / "trained" / skill_id
                model_path.mkdir(parents=True, exist_ok=True)
                self.model.save(str(model_path / "model.zip"))

                # Save the VecNormalize stats from the training environment
                try:
                    self.training_env.save(str(model_path / "vec_normalize.pkl"))
                except:
                    pass

                # Reload for viewing (in a thread-safe way)
                try:
                    new_policy = PPO.load(str(model_path / "model.zip"))
                    state["policy"] = new_policy
                    state["current_skill"] = skill_id

                    # Also reload normalization stats
                    import pickle
                    normalize_path = model_path / "vec_normalize.pkl"
                    if normalize_path.exists():
                        with open(normalize_path, 'rb') as f:
                            vec_normalize_data = pickle.load(f)
                        state["vec_normalize"] = {
                            "obs_rms_mean": vec_normalize_data.obs_rms.mean,
                            "obs_rms_var": vec_normalize_data.obs_rms.var,
                            "clip_obs": vec_normalize_data.clip_obs,
                            "epsilon": vec_normalize_data.epsilon,
                        }
                    print(f"[Training] Step {self.num_timesteps:,} - policy reloaded for viewing")
                except Exception as e:
                    print(f"[Training] Failed to reload policy: {e}")

            return True

    callback = LiveViewCallback()

    try:
        print(f"[Training] Starting {skill_id} for {timesteps:,} timesteps...")
        result = run_training(skill_id, timesteps, render=False, callback=callback)
        print(f"[Training] Complete! Result: {result}")

        # Load final trained policy
        load_skill(skill_id)

    except Exception as e:
        print(f"[Training] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state["training_active"] = False
        state["training_skill"] = None
        # Stop the robot when training ends
        state["policy"] = None
        state["vec_normalize"] = None
        state["current_skill"] = None
        # Reset the simulation to standing pose
        mujoco.mj_resetData(state["model"], state["data"])
        mujoco.mj_forward(state["model"], state["data"])
        state["step"] = 0
        print("[Training] Robot stopped and reset")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>G1 Robot Skill Learning</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            height: 100vh;
            overflow: hidden;
        }

        .app {
            display: grid;
            grid-template-columns: 1fr 400px;
            height: 100vh;
        }

        /* Left Panel - Simulation */
        .sim-panel {
            background: #161b22;
            padding: 20px;
            display: flex;
            flex-direction: column;
        }

        .sim-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .sim-header h1 {
            color: #58a6ff;
            font-size: 1.4em;
        }

        .skill-badge {
            background: #238636;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        .sim-container {
            flex: 1;
            background: #0d1117;
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }

        .sim-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 4px;
        }

        .stats-bar {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
        }

        .stat {
            background: #21262d;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }

        .stat-label {
            font-size: 0.75em;
            color: #8b949e;
            text-transform: uppercase;
        }

        .stat-value {
            font-size: 1.3em;
            color: #58a6ff;
            margin-top: 4px;
        }

        /* Right Panel - Chat */
        .chat-panel {
            background: #161b22;
            border-left: 1px solid #30363d;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 15px 20px;
            border-bottom: 1px solid #30363d;
        }

        .chat-header h2 {
            font-size: 1em;
            color: #c9d1d9;
        }

        .messages {
            flex: 1;
            overflow-y: scroll;
            padding: 15px;
            max-height: calc(100vh - 250px);
        }

        .message {
            margin-bottom: 15px;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            text-align: right;
        }

        .message.user .bubble {
            background: #238636;
            color: white;
        }

        .message.assistant .bubble {
            background: #21262d;
        }

        .bubble {
            display: inline-block;
            padding: 10px 14px;
            border-radius: 12px;
            max-width: 85%;
            text-align: left;
            font-size: 0.9em;
            line-height: 1.4;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        .bubble pre {
            background: #0d1117;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            margin: 8px 0;
            font-size: 0.85em;
        }

        .bubble code {
            font-family: 'SF Mono', Monaco, monospace;
        }

        .input-area {
            padding: 15px;
            border-top: 1px solid #30363d;
        }

        .input-container {
            display: flex;
            gap: 10px;
        }

        input[type="text"] {
            flex: 1;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            color: #c9d1d9;
            font-size: 0.9em;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: #58a6ff;
        }

        button {
            background: #238636;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }

        button:hover {
            background: #2ea043;
        }

        button:disabled {
            background: #21262d;
            cursor: not-allowed;
        }

        .quick-actions {
            display: flex;
            gap: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .quick-btn {
            background: #21262d;
            border: 1px solid #30363d;
            color: #8b949e;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            cursor: pointer;
        }

        .quick-btn:hover {
            border-color: #58a6ff;
            color: #58a6ff;
        }

        /* Training Progress */
        .training-bar {
            background: #21262d;
            padding: 10px 15px;
            border-bottom: 1px solid #30363d;
            display: none;
        }

        .training-bar.active {
            display: block;
        }

        .progress {
            height: 6px;
            background: #30363d;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            background: #238636;
            transition: width 0.3s;
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="app">
        <!-- Simulation Panel -->
        <div class="sim-panel">
            <div class="sim-header">
                <h1>Unitree G1 Humanoid</h1>
                <span class="skill-badge" id="skill-badge">No Skill</span>
            </div>

            <div class="sim-container">
                <img id="sim-frame" src="/video_feed" alt="Robot Simulation">
            </div>

            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-label">Height</div>
                    <div class="stat-value" id="stat-height">0.00m</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Velocity</div>
                    <div class="stat-value" id="stat-velocity">0.00</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Reward</div>
                    <div class="stat-value" id="stat-reward">0.00</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Step</div>
                    <div class="stat-value" id="stat-step">0</div>
                </div>
            </div>
        </div>

        <!-- Chat Panel -->
        <div class="chat-panel">
            <div class="chat-header">
                <h2>AI Training Assistant</h2>
            </div>

            <div class="training-bar" id="training-bar">
                <small>Training in progress...</small>
                <div class="progress">
                    <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                </div>
            </div>

            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="bubble">
                        Welcome! I'm your AI assistant for teaching the G1 robot new skills.

                        <strong>Available skills:</strong>
                        <ul>
                            <li>balance_stand - Basic balance</li>
                            <li>walk_forward - Forward walking</li>
                            <li>jump - Vertical jump</li>
                        </ul>

                        Try saying: "Train the robot to walk forward"
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <input type="text" id="user-input" placeholder="Tell me what to teach the robot..."
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()" id="send-btn">Send</button>
                </div>
                <div class="quick-actions">
                    <button class="quick-btn" onclick="quickAction('Train walk_forward 200k')">Train Walking</button>
                    <button class="quick-btn" onclick="quickAction('Train balance 100k')">Train Balance</button>
                    <button class="quick-btn" onclick="quickAction('Show the walking skill')">Show Walking</button>
                    <button class="quick-btn" onclick="quickAction('Reset simulation')">Reset</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;

        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;

            // Simple markdown-like formatting
            let formatted = content
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/\\n/g, '<br>');

            msgDiv.innerHTML = `<div class="bubble">${formatted}</div>`;
            messagesDiv.appendChild(msgDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();

            if (!message || isProcessing) return;

            isProcessing = true;
            document.getElementById('send-btn').disabled = true;

            addMessage('user', message);
            input.value = '';

            try {
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message})
                });
                const data = await resp.json();
                addMessage('assistant', data.response);

                // Update skill badge if changed
                if (data.skill) {
                    document.getElementById('skill-badge').textContent = data.skill;
                }
            } catch (e) {
                addMessage('assistant', 'Sorry, there was an error processing your request.');
            }

            isProcessing = false;
            document.getElementById('send-btn').disabled = false;
        }

        function quickAction(text) {
            document.getElementById('user-input').value = text;
            sendMessage();
        }

        // Update stats periodically
        setInterval(async () => {
            try {
                const resp = await fetch('/stats');
                const data = await resp.json();
                document.getElementById('stat-height').textContent = data.com_height.toFixed(2) + 'm';
                document.getElementById('stat-velocity').textContent = data.forward_vel.toFixed(2);
                document.getElementById('stat-reward').textContent = data.reward.toFixed(2);
                document.getElementById('stat-step').textContent = data.step;

                if (data.skill) {
                    document.getElementById('skill-badge').textContent = data.skill;
                }

                // Update training progress
                const trainingBar = document.getElementById('training-bar');
                const progressFill = document.getElementById('progress-fill');
                if (data.training_active) {
                    trainingBar.classList.add('active');
                    const pct = (data.training_progress / data.training_total * 100).toFixed(1);
                    progressFill.style.width = pct + '%';
                    trainingBar.querySelector('small').textContent =
                        `Training ${data.training_skill}: ${(data.training_progress/1000).toFixed(0)}k / ${(data.training_total/1000).toFixed(0)}k steps (${pct}%)`;
                } else {
                    trainingBar.classList.remove('active');
                }
            } catch (e) {}
        }, 200);

        // Fallback: If MJPEG doesn't load within 3 seconds, switch to polling
        let mjpegFailed = true;
        const simImg = document.getElementById('sim-frame');

        simImg.onload = () => { mjpegFailed = false; };
        simImg.onerror = () => { startPolling(); };

        setTimeout(() => {
            if (mjpegFailed) {
                console.log('MJPEG stream not loading, switching to polling mode');
                startPolling();
            }
        }, 3000);

        function startPolling() {
            simImg.src = '';  // Clear MJPEG stream
            setInterval(() => {
                // Add timestamp to prevent caching
                simImg.src = '/frame?' + Date.now();
            }, 100);  // 10 FPS polling
        }
    </script>
</body>
</html>
"""


def init_simulation():
    """Initialize MuJoCo simulation (model and data only)."""
    # Use scene file which includes ground plane
    xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
    state["model"] = mujoco.MjModel.from_xml_path(str(xml_path))
    state["data"] = mujoco.MjData(state["model"])
    mujoco.mj_forward(state["model"], state["data"])
    print("MuJoCo simulation initialized")


def init_renderer():
    """Initialize renderer (must be called from the thread that will use it)."""
    try:
        state["renderer"] = mujoco.Renderer(state["model"], height=720, width=1280)
        print("MuJoCo renderer initialized")
    except Exception as e:
        print(f"Error initializing renderer: {e}")
        raise


def init_orchestrator():
    """Initialize the AI orchestrator."""
    state["orchestrator"] = UnifiedOrchestrator()
    print("Orchestrator initialized")


def load_skill(skill_id: str):
    """Load a trained skill policy and normalization stats."""
    skill_dir = PROJECT_ROOT / "skills" / "trained" / skill_id
    model_path = skill_dir / "model.zip"
    normalize_path = skill_dir / "vec_normalize.pkl"

    if model_path.exists():
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        import pickle

        state["policy"] = PPO.load(str(model_path))
        state["current_skill"] = skill_id

        # Load normalization statistics if available
        if normalize_path.exists():
            try:
                with open(normalize_path, 'rb') as f:
                    vec_normalize_data = pickle.load(f)
                # Extract the observation normalization parameters
                state["vec_normalize"] = {
                    "obs_rms_mean": vec_normalize_data.obs_rms.mean,
                    "obs_rms_var": vec_normalize_data.obs_rms.var,
                    "clip_obs": vec_normalize_data.clip_obs,
                    "epsilon": vec_normalize_data.epsilon,
                }
                print(f"Loaded normalization stats for: {skill_id}")
            except Exception as e:
                print(f"Warning: Could not load normalization stats: {e}")
                state["vec_normalize"] = None
        else:
            print(f"Warning: No normalization stats found for {skill_id}")
            state["vec_normalize"] = None

        print(f"Loaded skill: {skill_id}")
        return True
    return False


def get_observation():
    """Get observation vector."""
    data = state["data"]
    qpos = data.qpos[3:].copy()
    qvel = data.qvel.copy()
    torso_xmat = data.xmat[1].reshape(9)
    return np.concatenate([qpos, qvel, torso_xmat])


def normalize_observation(obs):
    """Normalize observation using stored VecNormalize statistics."""
    vec_norm = state.get("vec_normalize")
    if vec_norm is None:
        return obs

    # Apply the same normalization as VecNormalize
    mean = vec_norm["obs_rms_mean"]
    var = vec_norm["obs_rms_var"]
    epsilon = vec_norm["epsilon"]
    clip_obs = vec_norm["clip_obs"]

    normalized = (obs - mean) / np.sqrt(var + epsilon)
    normalized = np.clip(normalized, -clip_obs, clip_obs)
    return normalized


def step_simulation():
    """Step the simulation."""
    model = state["model"]
    data = state["data"]
    policy = state["policy"]

    if policy is not None:
        obs = get_observation()
        # Normalize observation to match training distribution
        obs_normalized = normalize_observation(obs)
        action, _ = policy.predict(obs_normalized, deterministic=True)
        data.ctrl[:] = action * model.actuator_ctrlrange[:, 1]
    else:
        data.ctrl[:] = 0

    for _ in range(5):
        mujoco.mj_step(model, data)

    state["step"] += 1


def create_placeholder_frame():
    """Create a placeholder frame while simulation loads."""
    img = Image.new('RGB', (1280, 720), color=(22, 27, 34))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=80)
    return buffer.getvalue()


def render_frame():
    """Render current frame as JPEG."""
    renderer = state.get("renderer")
    data = state["data"]

    if renderer is None:
        return create_placeholder_frame()

    # Set up camera to look at the robot
    camera = state.get("camera")
    if camera is None:
        camera = mujoco.MjvCamera()
        camera.azimuth = 135
        camera.elevation = -20
        camera.distance = 3.0
        camera.lookat[:] = [0, 0, 0.8]
        state["camera"] = camera

    # Track the robot
    camera.lookat[0] = data.qpos[0]
    camera.lookat[1] = data.qpos[1]

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=80)
    return buffer.getvalue()


def generate_frames():
    """Generator for video stream - reads from frame buffer."""
    placeholder = create_placeholder_frame()

    while state["running"]:
        with state["frame_lock"]:
            frame = state["latest_frame"]

        if frame is None:
            frame = placeholder

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1/15)  # 15fps to reduce CPU load


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/frame')
def single_frame():
    """Get a single frame (fallback for MJPEG issues)."""
    with state["frame_lock"]:
        frame = state["latest_frame"]

    if frame is None:
        frame = create_placeholder_frame()

    return Response(frame, mimetype='image/jpeg')


@app.route('/stats')
def stats():
    """Return simulation stats."""
    return jsonify({
        "step": state.get("step", 0),
        "com_height": state.get("com_height", 0),
        "forward_vel": state.get("forward_vel", 0),
        "reward": state.get("reward", 0),
        "skill": state.get("current_skill"),
        "training_active": state.get("training_active", False),
        "training_skill": state.get("training_skill"),
        "training_progress": state.get("training_timesteps", 0),
        "training_total": state.get("training_total", 0),
    })


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    data = request.json
    message = data.get('message', '')

    # Check for skill loading commands
    lower_msg = message.lower()

    if 'show' in lower_msg and 'balance' in lower_msg:
        if load_skill('balance_stand'):
            return jsonify({
                "response": "Loaded the **balance_stand** skill. Watch the robot maintain its balance in the simulation!",
                "skill": "balance_stand"
            })
        else:
            return jsonify({
                "response": "The balance skill hasn't been trained yet. Would you like me to train it?",
                "skill": None
            })

    if 'show' in lower_msg and 'walk' in lower_msg:
        if load_skill('walk_forward'):
            return jsonify({
                "response": "Loaded the **walk_forward** skill. Watch the robot walk in the simulation!",
                "skill": "walk_forward"
            })
        else:
            return jsonify({
                "response": "The walking skill hasn't been trained yet. Train `balance_stand` first, then `walk_forward`.",
                "skill": None
            })

    if 'reset' in lower_msg:
        mujoco.mj_resetData(state["model"], state["data"])
        mujoco.mj_forward(state["model"], state["data"])
        state["step"] = 0
        state["policy"] = None  # Clear policy so robot stays still
        state["vec_normalize"] = None  # Clear normalization
        state["current_skill"] = None
        return jsonify({
            "response": "Simulation reset. The robot is standing still. Load a skill to see it move.",
            "skill": None
        })

    if 'list' in lower_msg and 'skill' in lower_msg:
        skills_list = "\n".join([f"- **{k}**: {v['description']}" for k, v in SKILL_TEMPLATES.items()])
        return jsonify({
            "response": f"**Available Skills:**\n\n{skills_list}\n\nTo train: Say 'train walk_forward'\nTo view: Say 'show walking skill'",
            "skill": state.get("current_skill")
        })

    # Handle training commands
    if 'train' in lower_msg or ('yes' in lower_msg and 'train' in message.lower()):
        if state["training_active"]:
            progress = state["training_timesteps"]
            total = state["training_total"]
            pct = (progress / total * 100) if total > 0 else 0
            return jsonify({
                "response": f"Training **{state['training_skill']}** is already in progress: {progress:,}/{total:,} steps ({pct:.1f}%)",
                "skill": state.get("current_skill")
            })

        # Parse which skill to train
        skill_to_train = None
        timesteps = 100000  # Default

        # Check predefined skills
        if 'walk' in lower_msg:
            skill_to_train = 'walk_forward'
            timesteps = 200000
        elif 'balance' in lower_msg:
            skill_to_train = 'balance_stand'
            timesteps = 100000
        elif 'jump' in lower_msg:
            skill_to_train = 'jump'
            timesteps = 300000
        elif 'raise' in lower_msg or 'hand' in lower_msg:
            skill_to_train = 'raise_hand'
            timesteps = 500000
        elif 'wave' in lower_msg:
            skill_to_train = 'wave'
            timesteps = 300000

        # Check for custom skill configs
        if skill_to_train is None:
            configs_dir = PROJECT_ROOT / "skills" / "configs"
            if configs_dir.exists():
                for config_file in configs_dir.glob("*.json"):
                    skill_name = config_file.stem
                    if skill_name.replace('_', ' ') in lower_msg or skill_name in lower_msg:
                        skill_to_train = skill_name
                        # Load timesteps from config
                        try:
                            import json as json_module
                            with open(config_file) as f:
                                config = json_module.load(f)
                                timesteps = config.get('training_timesteps', 100000)
                        except:
                            pass
                        break

        # Check for custom timesteps in message
        import re
        ts_match = re.search(r'(\d+)\s*k?\s*(steps|timesteps)?', lower_msg)
        if ts_match:
            num = int(ts_match.group(1))
            if 'k' in lower_msg[ts_match.start():ts_match.end()+2]:
                num *= 1000
            if num >= 1000:
                timesteps = num

        if skill_to_train:
            # Check if resuming from previous training
            prev_model_path = PROJECT_ROOT / "skills" / "trained" / skill_to_train / "model.zip"
            prev_metrics_path = PROJECT_ROOT / "skills" / "trained" / skill_to_train / "metrics.json"

            resuming = prev_model_path.exists()
            prev_timesteps = 0
            if resuming and prev_metrics_path.exists():
                try:
                    import json as json_module
                    with open(prev_metrics_path) as f:
                        prev_metrics = json_module.load(f)
                        prev_timesteps = prev_metrics.get("cumulative_timesteps", 0)
                except:
                    pass

            # Start training in background
            t = threading.Thread(target=training_thread, args=(skill_to_train, timesteps), daemon=True)
            t.start()

            if resuming:
                response_msg = f"**Resuming** training for **{skill_to_train}** from {prev_timesteps:,} timesteps (+{timesteps:,} more). Watch the simulation - the policy will update live as training progresses."
            else:
                response_msg = f"Started training **{skill_to_train}** for {timesteps:,} timesteps! Watch the simulation - the policy will update live as training progresses."

            return jsonify({
                "response": response_msg,
                "skill": skill_to_train
            })
        else:
            # List available skills
            available = list(SKILL_TEMPLATES.keys())
            configs_dir = PROJECT_ROOT / "skills" / "configs"
            if configs_dir.exists():
                available.extend([f.stem for f in configs_dir.glob("*.json")])
            skills_str = ", ".join(available)
            return jsonify({
                "response": f"Which skill would you like to train? Available: {skills_str}",
                "skill": state.get("current_skill")
            })

    # Pass to orchestrator for complex queries
    try:
        response = state["orchestrator"].process(message)

        # Check if orchestrator wants to start training
        import re
        training_match = re.search(r'\[START_TRAINING:(\w+):(\d+)\]', response)
        if training_match and not state["training_active"]:
            skill_id = training_match.group(1)
            timesteps = int(training_match.group(2))

            # Start training in background
            t = threading.Thread(target=training_thread, args=(skill_id, timesteps), daemon=True)
            t.start()

            # Remove the marker from response
            response = re.sub(r'\[START_TRAINING:\w+:\d+\]\s*', '', response)

            return jsonify({
                "response": response,
                "skill": skill_id
            })

        return jsonify({
            "response": response,
            "skill": state.get("current_skill")
        })
    except Exception as e:
        return jsonify({
            "response": f"Error: {str(e)}",
            "skill": None
        })


@app.route('/load_skill/<skill_id>')
def load_skill_route(skill_id):
    """Load a specific skill."""
    if load_skill(skill_id):
        return jsonify({"success": True, "skill": skill_id})
    return jsonify({"success": False, "error": "Skill not found"})


def run_flask(port):
    """Run Flask server in a background thread."""
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False, use_reloader=False)


def main_simulation_loop():
    """Main thread simulation loop with rendering."""
    print("Simulation loop running on main thread")
    reset_delay = 0  # Frames to wait before resetting after fall

    while state["running"]:
        try:
            step_simulation()

            # Update stats
            data = state["data"]
            com_height = float(data.subtree_com[0][2])
            state["com_height"] = com_height
            state["forward_vel"] = float(data.qvel[0])

            # Check if robot is upright
            torso_xmat = data.xmat[1].reshape(3, 3)
            upright = torso_xmat[2, 2]  # z-component of z-axis (1.0 = perfectly upright)
            state["reward"] = upright + com_height

            # Auto-reset if robot fell or tilted too much
            robot_fell = com_height < 0.4 or upright < 0.3

            if robot_fell:
                reset_delay += 1
                # Wait a bit so user can see the fall, then reset
                if reset_delay > 15:  # ~1 second at 15fps
                    mujoco.mj_resetData(state["model"], state["data"])
                    mujoco.mj_forward(state["model"], state["data"])
                    state["step"] = 0
                    reset_delay = 0
                    print("[Viewer] Robot fell - auto reset")
            else:
                reset_delay = 0

            # Render and store frame
            frame = render_frame()
            with state["frame_lock"]:
                state["latest_frame"] = frame

            time.sleep(1/15)  # 15fps to reduce CPU load
        except KeyboardInterrupt:
            state["running"] = False
            break
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G1 Robot Web App")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on")
    parser.add_argument("--skill", type=str, help="Initial skill to load")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("G1 Robot Skill Learning System")
    print("="*50)

    # Initialize simulation and renderer on main thread (required for macOS OpenGL)
    print("\nInitializing...")
    init_simulation()
    init_renderer()
    init_orchestrator()

    if args.skill:
        load_skill(args.skill)

    # Start Flask in background thread
    print(f"\nStarting web server on http://localhost:{args.port}")
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()

    print("="*50)
    print("Open the URL above in your browser")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")

    # Run simulation on main thread (required for OpenGL on macOS)
    try:
        main_simulation_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        state["running"] = False


if __name__ == "__main__":
    main()
