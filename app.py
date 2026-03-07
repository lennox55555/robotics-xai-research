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

app = Flask(__name__)

# Global state
state = {
    "orchestrator": None,
    "model": None,
    "data": None,
    "renderer": None,
    "policy": None,
    "current_skill": None,
    "running": True,
    "step": 0,
    "messages": [],
    "training_active": False,
    "training_progress": 0,
}

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
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .sim-container img {
            max-width: 100%;
            max-height: 100%;
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
            overflow-y: auto;
            padding: 15px;
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
                    <button class="quick-btn" onclick="quickAction('List available skills')">List Skills</button>
                    <button class="quick-btn" onclick="quickAction('Show the balance skill')">Show Balance</button>
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
            } catch (e) {}
        }, 200);
    </script>
</body>
</html>
"""


def init_simulation():
    """Initialize MuJoCo simulation."""
    xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
    state["model"] = mujoco.MjModel.from_xml_path(str(xml_path))
    state["data"] = mujoco.MjData(state["model"])
    state["renderer"] = mujoco.Renderer(state["model"], height=480, width=640)
    mujoco.mj_forward(state["model"], state["data"])
    print("MuJoCo simulation initialized")


def init_orchestrator():
    """Initialize the AI orchestrator."""
    state["orchestrator"] = UnifiedOrchestrator()
    print("Orchestrator initialized")


def load_skill(skill_id: str):
    """Load a trained skill policy."""
    model_path = PROJECT_ROOT / "skills" / "trained" / skill_id / "model.zip"
    if model_path.exists():
        from stable_baselines3 import PPO
        state["policy"] = PPO.load(str(model_path))
        state["current_skill"] = skill_id
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


def step_simulation():
    """Step the simulation."""
    model = state["model"]
    data = state["data"]
    policy = state["policy"]

    if policy is not None:
        obs = get_observation()
        action, _ = policy.predict(obs, deterministic=True)
        data.ctrl[:] = action * model.actuator_ctrlrange[:, 1]
    else:
        data.ctrl[:] = 0

    for _ in range(5):
        mujoco.mj_step(model, data)

    state["step"] += 1


def render_frame():
    """Render current frame as JPEG."""
    renderer = state["renderer"]
    data = state["data"]

    renderer.update_scene(data)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=80)
    return buffer.getvalue()


def generate_frames():
    """Generator for video stream."""
    while state["running"]:
        step_simulation()

        # Update stats
        data = state["data"]
        state["com_height"] = float(data.subtree_com[0][2])
        state["forward_vel"] = float(data.qvel[0])
        upright = data.xmat[1].reshape(3, 3)[2, 2]
        state["reward"] = upright + state["com_height"]

        frame = render_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1/30)


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


@app.route('/stats')
def stats():
    """Return simulation stats."""
    return jsonify({
        "step": state.get("step", 0),
        "com_height": state.get("com_height", 0),
        "forward_vel": state.get("forward_vel", 0),
        "reward": state.get("reward", 0),
        "skill": state.get("current_skill"),
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
        return jsonify({
            "response": "Simulation reset. The robot is back to its initial pose.",
            "skill": state.get("current_skill")
        })

    if 'list' in lower_msg and 'skill' in lower_msg:
        skills_list = "\n".join([f"- **{k}**: {v['description']}" for k, v in SKILL_TEMPLATES.items()])
        return jsonify({
            "response": f"**Available Skills:**\n\n{skills_list}\n\nTo train: `python run_orchestrator.py --train <skill_id>`\nTo view: Say 'Show the walking skill'",
            "skill": state.get("current_skill")
        })

    # Pass to orchestrator for complex queries
    try:
        response = state["orchestrator"].process(message)
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="G1 Robot Web App")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on")
    parser.add_argument("--skill", type=str, help="Initial skill to load")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("G1 Robot Skill Learning System")
    print("="*50)

    # Initialize
    print("\nInitializing...")
    init_simulation()
    init_orchestrator()

    if args.skill:
        load_skill(args.skill)

    print(f"\nStarting web server...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
