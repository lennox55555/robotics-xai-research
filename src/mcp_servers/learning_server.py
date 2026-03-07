"""
Training Agent MCP Server

Provides tools for training new skills:
- Train a skill from scratch
- Train with transfer learning
- Monitor training progress
- Evaluate trained skills
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.skill_learning.skill import Skill, SkillConfig, SkillStatus, SkillLibrary
from src.skill_learning.skill_trainer import SkillTrainer

# Server instance
server = Server("learning-agent")

# Global state
_skills_dir = Path(__file__).parent.parent.parent / "skills"
_skill_library = None
_trainer = None
_training_status: Dict[str, Dict] = {}  # Track ongoing training


def get_skill_library() -> SkillLibrary:
    global _skill_library
    if _skill_library is None:
        _skill_library = SkillLibrary(_skills_dir)
    return _skill_library


def get_trainer() -> SkillTrainer:
    global _trainer
    if _trainer is None:
        from src.envs.g1_humanoid import G1HumanoidEnv
        _trainer = SkillTrainer(G1HumanoidEnv, _skills_dir)
    return _trainer


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available training tools."""
    return [
        Tool(
            name="create_skill",
            description="Create a new skill definition (does not train it)",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "Unique ID for the skill (snake_case)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this skill does"
                    },
                    "reward_components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Reward function components"
                    },
                    "success_criteria": {
                        "type": "string",
                        "description": "How to measure success"
                    },
                    "prerequisites": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skill IDs that must be trained first"
                    },
                    "transfer_from": {
                        "type": "string",
                        "description": "Skill ID to transfer weights from"
                    }
                },
                "required": ["skill_id", "name", "description", "reward_components"]
            }
        ),
        Tool(
            name="train_skill",
            description="Start training a skill using reinforcement learning",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to train"
                    },
                    "total_timesteps": {
                        "type": "integer",
                        "description": "Total training timesteps",
                        "default": 500000
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["PPO", "SAC", "TD3"],
                        "description": "RL algorithm to use",
                        "default": "PPO"
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="get_training_status",
            description="Get the status of ongoing or completed training",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to check"
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="list_skills",
            description="List all skills (pending, training, trained)",
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "pending", "training", "trained", "failed"],
                        "description": "Filter by status"
                    }
                }
            }
        ),
        Tool(
            name="evaluate_skill",
            description="Evaluate a trained skill's performance",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to evaluate"
                    },
                    "n_episodes": {
                        "type": "integer",
                        "description": "Number of evaluation episodes",
                        "default": 10
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="get_skill_details",
            description="Get detailed information about a skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill"
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="suggest_transfer_source",
            description="Suggest which existing skill to transfer from for a new skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_skill_id": {
                        "type": "string",
                        "description": "ID of the skill that needs training"
                    }
                },
                "required": ["target_skill_id"]
            }
        ),
        Tool(
            name="get_reward_components",
            description="List available reward components for skill training",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    library = get_skill_library()

    if name == "create_skill":
        config = SkillConfig(
            transfer_from=arguments.get("transfer_from"),
            total_timesteps=arguments.get("total_timesteps", 500000),
            algorithm=arguments.get("algorithm", "PPO"),
        )

        skill = Skill(
            skill_id=arguments["skill_id"],
            name=arguments["name"],
            description=arguments["description"],
            reward_components=arguments["reward_components"],
            success_criteria=arguments.get("success_criteria", "Task completed"),
            termination_conditions=arguments.get("termination_conditions", ["robot_fell"]),
            prerequisites=arguments.get("prerequisites", []),
            config=config,
        )

        library.add_skill(skill)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "created",
                "skill_id": skill.skill_id,
                "name": skill.name,
                "reward_components": skill.reward_components,
            })
        )]

    elif name == "train_skill":
        skill_id = arguments["skill_id"]
        skill = library.get_skill(skill_id)

        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        # Update config if provided
        if "total_timesteps" in arguments:
            skill.config.total_timesteps = arguments["total_timesteps"]
        if "algorithm" in arguments:
            skill.config.algorithm = arguments["algorithm"]

        # Start training in background
        _training_status[skill_id] = {
            "status": "starting",
            "progress": 0,
        }

        # Note: In production, this would be async/threaded
        # For now, we'll do synchronous training
        try:
            trainer = get_trainer()
            metrics = trainer.train(skill)

            _training_status[skill_id] = {
                "status": "completed",
                "metrics": metrics,
            }

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "training_complete",
                    "skill_id": skill_id,
                    "metrics": metrics,
                })
            )]

        except Exception as e:
            _training_status[skill_id] = {
                "status": "failed",
                "error": str(e),
            }
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "training_failed",
                    "skill_id": skill_id,
                    "error": str(e),
                })
            )]

    elif name == "get_training_status":
        skill_id = arguments["skill_id"]

        if skill_id in _training_status:
            return [TextContent(
                type="text",
                text=json.dumps(_training_status[skill_id])
            )]

        skill = library.get_skill(skill_id)
        if skill:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": skill.status.value,
                    "metrics": skill.training_metrics,
                })
            )]

        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Skill '{skill_id}' not found"})
        )]

    elif name == "list_skills":
        status_filter = arguments.get("status_filter", "all")

        skills = []
        for skill_id, skill in library.skills.items():
            if status_filter == "all" or skill.status.value == status_filter:
                skills.append({
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "status": skill.status.value,
                    "description": skill.description,
                })

        return [TextContent(
            type="text",
            text=json.dumps({
                "skills": skills,
                "count": len(skills),
            })
        )]

    elif name == "evaluate_skill":
        skill_id = arguments["skill_id"]
        n_episodes = arguments.get("n_episodes", 10)

        skill = library.get_skill(skill_id)
        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        if skill.status != SkillStatus.TRAINED:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' is not trained"})
            )]

        trainer = get_trainer()
        results = trainer.evaluate(skill, n_episodes=n_episodes)

        return [TextContent(
            type="text",
            text=json.dumps({
                "skill_id": skill_id,
                "evaluation": results,
            })
        )]

    elif name == "get_skill_details":
        skill_id = arguments["skill_id"]
        skill = library.get_skill(skill_id)

        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        return [TextContent(
            type="text",
            text=json.dumps(skill.to_dict())
        )]

    elif name == "suggest_transfer_source":
        skill_id = arguments["target_skill_id"]
        skill = library.get_skill(skill_id)

        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        similar = library.find_similar_skills(skill)

        suggestions = [
            {
                "skill_id": s.skill_id,
                "name": s.name,
                "reward_components": s.reward_components,
            }
            for s in similar
        ]

        return [TextContent(
            type="text",
            text=json.dumps({
                "target_skill": skill_id,
                "suggestions": suggestions,
            })
        )]

    elif name == "get_reward_components":
        components = {
            "height_reward": "Reward for maintaining/achieving height",
            "upright_reward": "Reward for staying upright (torso orientation)",
            "velocity_forward": "Reward for forward movement",
            "velocity_target": "Reward for matching target velocity",
            "energy_efficiency": "Penalty for excessive control effort",
            "stability": "Reward for low joint velocities",
            "foot_contact": "Reward for proper foot contact patterns",
            "joint_limits": "Penalty for approaching joint limits",
            "symmetry": "Reward for symmetric movements",
            "smoothness": "Penalty for jerky movements",
        }

        return [TextContent(
            type="text",
            text=json.dumps({
                "available_components": components,
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
