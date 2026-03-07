"""
Physical World MCP Server

Provides tools for interacting with the simulated robot:
- Run simulations
- Execute skills
- Get sensor data
- Observe robot state
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO

# Server instance
server = Server("performance-agent")

# Global state
_env = None
_current_obs = None
_loaded_skills: Dict[str, Any] = {}
_skills_dir = Path(__file__).parent.parent.parent / "skills"


def get_env():
    """Get or create the simulation environment."""
    global _env, _current_obs
    if _env is None:
        from src.envs.g1_humanoid import G1HumanoidEnv
        _env = G1HumanoidEnv()
        _current_obs, _ = _env.reset()
    return _env


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available physical world tools."""
    return [
        Tool(
            name="reset_simulation",
            description="Reset the robot simulation to initial state",
            inputSchema={
                "type": "object",
                "properties": {
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reset"
                    }
                }
            }
        ),
        Tool(
            name="step_simulation",
            description="Step the simulation with a random or specified action",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Action array (43 values for G1). If not provided, uses random action."
                    },
                    "n_steps": {
                        "type": "integer",
                        "description": "Number of steps to take",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="get_robot_state",
            description="Get current state of the robot (joint positions, velocities, COM)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="execute_skill",
            description="Execute a trained skill on the robot",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to execute"
                    },
                    "n_steps": {
                        "type": "integer",
                        "description": "Number of steps to run the skill",
                        "default": 100
                    },
                    "deterministic": {
                        "type": "boolean",
                        "description": "Use deterministic policy",
                        "default": True
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="list_available_skills",
            description="List all trained skills available for execution",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="run_skill_episode",
            description="Run a complete episode using a skill until termination",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to run"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum steps per episode",
                        "default": 1000
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="get_joint_info",
            description="Get information about robot joints and actuators",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global _current_obs

    if name == "reset_simulation":
        env = get_env()
        seed = arguments.get("seed")
        _current_obs, info = env.reset(seed=seed)
        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "reset",
                "observation_shape": list(_current_obs.shape),
                "com_height": float(info.get("com_height", 0)),
            })
        )]

    elif name == "step_simulation":
        env = get_env()
        n_steps = arguments.get("n_steps", 1)
        action = arguments.get("action")

        total_reward = 0
        for _ in range(n_steps):
            if action is None:
                act = env.action_space.sample()
            else:
                act = np.array(action, dtype=np.float32)

            _current_obs, reward, terminated, truncated, info = env.step(act)
            total_reward += reward

            if terminated or truncated:
                break

        return [TextContent(
            type="text",
            text=json.dumps({
                "steps_taken": n_steps,
                "total_reward": float(total_reward),
                "terminated": terminated,
                "truncated": truncated,
                "com_height": float(info.get("com_height", 0)),
            })
        )]

    elif name == "get_robot_state":
        env = get_env()
        if _current_obs is None:
            _current_obs, _ = env.reset()

        return [TextContent(
            type="text",
            text=json.dumps({
                "observation": _current_obs.tolist()[:20],  # First 20 values
                "observation_size": len(_current_obs),
                "com_position": env.data.subtree_com[0].tolist(),
                "com_height": float(env.data.subtree_com[0][2]),
            })
        )]

    elif name == "execute_skill":
        skill_id = arguments["skill_id"]
        n_steps = arguments.get("n_steps", 100)
        deterministic = arguments.get("deterministic", True)

        # Load skill if not loaded
        if skill_id not in _loaded_skills:
            model_path = _skills_dir / "trained" / skill_id / "model.zip"
            if not model_path.exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Skill '{skill_id}' not found"})
                )]
            _loaded_skills[skill_id] = PPO.load(str(model_path))

        model = _loaded_skills[skill_id]
        env = get_env()

        total_reward = 0
        for _ in range(n_steps):
            action, _ = model.predict(_current_obs, deterministic=deterministic)
            _current_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return [TextContent(
            type="text",
            text=json.dumps({
                "skill_id": skill_id,
                "steps_executed": n_steps,
                "total_reward": float(total_reward),
                "terminated": terminated,
                "truncated": truncated,
                "com_height": float(info.get("com_height", 0)),
            })
        )]

    elif name == "list_available_skills":
        trained_dir = _skills_dir / "trained"
        skills = []
        if trained_dir.exists():
            for skill_dir in trained_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "model.zip").exists():
                    skills.append(skill_dir.name)

        return [TextContent(
            type="text",
            text=json.dumps({
                "available_skills": skills,
                "count": len(skills)
            })
        )]

    elif name == "run_skill_episode":
        skill_id = arguments["skill_id"]
        max_steps = arguments.get("max_steps", 1000)

        # Load skill
        if skill_id not in _loaded_skills:
            model_path = _skills_dir / "trained" / skill_id / "model.zip"
            if not model_path.exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Skill '{skill_id}' not found"})
                )]
            _loaded_skills[skill_id] = PPO.load(str(model_path))

        model = _loaded_skills[skill_id]
        env = get_env()

        _current_obs, _ = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action, _ = model.predict(_current_obs, deterministic=True)
            _current_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        return [TextContent(
            type="text",
            text=json.dumps({
                "skill_id": skill_id,
                "episode_steps": steps,
                "episode_reward": float(total_reward),
                "terminated": terminated,
                "truncated": truncated,
                "final_com_height": float(info.get("com_height", 0)),
            })
        )]

    elif name == "get_joint_info":
        env = get_env()
        joint_names = [env.model.joint(i).name for i in range(env.model.njnt)]

        return [TextContent(
            type="text",
            text=json.dumps({
                "num_joints": env.model.njnt,
                "num_actuators": env.model.nu,
                "joint_names": joint_names,
            })
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
