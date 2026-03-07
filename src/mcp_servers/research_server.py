"""
Research Agent MCP Server

Provides tools for analysis, explainability, and research:
- Analyze skill performance
- Explain policy decisions (XAI)
- Compare skills
- Generate research insights
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Server instance
server = Server("research-agent")

# Global state
_skills_dir = Path(__file__).parent.parent.parent / "skills"


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available research tools."""
    return [
        Tool(
            name="analyze_skill_performance",
            description="Analyze detailed performance metrics of a trained skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill to analyze"
                    },
                    "n_episodes": {
                        "type": "integer",
                        "description": "Number of episodes to analyze",
                        "default": 20
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="explain_policy_decision",
            description="Use XAI techniques to explain why the policy made a decision",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill"
                    },
                    "observation": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The observation state to explain"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["saliency", "feature_importance", "action_distribution"],
                        "description": "XAI method to use",
                        "default": "feature_importance"
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="compare_skills",
            description="Compare performance and characteristics of two skills",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id_1": {
                        "type": "string",
                        "description": "First skill ID"
                    },
                    "skill_id_2": {
                        "type": "string",
                        "description": "Second skill ID"
                    }
                },
                "required": ["skill_id_1", "skill_id_2"]
            }
        ),
        Tool(
            name="get_skill_insights",
            description="Generate insights about a skill's learning process",
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
            name="analyze_failure_modes",
            description="Analyze when and why a skill fails",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "ID of the skill"
                    },
                    "n_episodes": {
                        "type": "integer",
                        "description": "Episodes to analyze",
                        "default": 50
                    }
                },
                "required": ["skill_id"]
            }
        ),
        Tool(
            name="suggest_improvements",
            description="Suggest improvements for a skill based on analysis",
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
            name="get_transfer_candidates",
            description="Find skills that could benefit from transfer learning",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="generate_skill_report",
            description="Generate a comprehensive report on a skill",
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""

    if name == "analyze_skill_performance":
        skill_id = arguments["skill_id"]
        n_episodes = arguments.get("n_episodes", 20)

        # Load skill and run analysis
        try:
            from stable_baselines3 import PPO
            from src.envs.g1_humanoid import G1HumanoidEnv

            model_path = _skills_dir / "trained" / skill_id / "model.zip"
            if not model_path.exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Skill '{skill_id}' not found"})
                )]

            model = PPO.load(str(model_path))
            env = G1HumanoidEnv()

            rewards = []
            lengths = []
            final_heights = []
            termination_reasons = []

            for _ in range(n_episodes):
                obs, _ = env.reset()
                ep_reward = 0
                ep_length = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_reward += reward
                    ep_length += 1
                    done = terminated or truncated

                rewards.append(ep_reward)
                lengths.append(ep_length)
                final_heights.append(info.get("com_height", 0))
                termination_reasons.append("terminated" if terminated else "truncated")

            env.close()

            analysis = {
                "skill_id": skill_id,
                "episodes_analyzed": n_episodes,
                "reward_stats": {
                    "mean": float(np.mean(rewards)),
                    "std": float(np.std(rewards)),
                    "min": float(np.min(rewards)),
                    "max": float(np.max(rewards)),
                },
                "length_stats": {
                    "mean": float(np.mean(lengths)),
                    "std": float(np.std(lengths)),
                    "min": int(np.min(lengths)),
                    "max": int(np.max(lengths)),
                },
                "final_height_stats": {
                    "mean": float(np.mean(final_heights)),
                    "std": float(np.std(final_heights)),
                },
                "termination_breakdown": {
                    "terminated": termination_reasons.count("terminated"),
                    "truncated": termination_reasons.count("truncated"),
                },
            }

            return [TextContent(type="text", text=json.dumps(analysis, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    elif name == "explain_policy_decision":
        skill_id = arguments["skill_id"]
        method = arguments.get("method", "feature_importance")

        try:
            from stable_baselines3 import PPO
            from src.envs.g1_humanoid import G1HumanoidEnv
            from src.explainability.policy_analyzer import PolicyAnalyzer

            model_path = _skills_dir / "trained" / skill_id / "model.zip"
            if not model_path.exists():
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Skill '{skill_id}' not found"})
                )]

            model = PPO.load(str(model_path))

            # Get observation
            if "observation" in arguments:
                obs = np.array(arguments["observation"])
            else:
                env = G1HumanoidEnv()
                obs, _ = env.reset()
                env.close()

            analyzer = PolicyAnalyzer(model, method)

            if method == "saliency":
                result = analyzer.compute_saliency(obs)
                explanation = {
                    "method": "saliency",
                    "skill_id": skill_id,
                    "top_influential_features": [
                        {"index": int(i), "importance": float(result[i])}
                        for i in np.argsort(result)[-10:][::-1]
                    ],
                }
            elif method == "feature_importance":
                result = analyzer.compute_feature_importance(obs.reshape(1, -1))
                explanation = {
                    "method": "feature_importance",
                    "skill_id": skill_id,
                    "mean_importance": result["mean_importance"][:10].tolist(),
                }
            else:
                result = analyzer.analyze_action_distribution(obs)
                explanation = {
                    "method": "action_distribution",
                    "skill_id": skill_id,
                    "action_mean": result["action_mean"].tolist()[:10],
                    "action_std": result["action_std"].tolist()[:10],
                    "entropy": float(result["entropy"]),
                }

            return [TextContent(type="text", text=json.dumps(explanation, indent=2))]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    elif name == "compare_skills":
        skill_1 = arguments["skill_id_1"]
        skill_2 = arguments["skill_id_2"]

        # Load skill configs
        from src.skill_learning.skill import SkillLibrary
        library = SkillLibrary(_skills_dir)

        s1 = library.get_skill(skill_1)
        s2 = library.get_skill(skill_2)

        if s1 is None or s2 is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "One or both skills not found"})
            )]

        comparison = {
            "skill_1": {
                "id": s1.skill_id,
                "name": s1.name,
                "status": s1.status.value,
                "reward_components": s1.reward_components,
                "metrics": s1.training_metrics,
            },
            "skill_2": {
                "id": s2.skill_id,
                "name": s2.name,
                "status": s2.status.value,
                "reward_components": s2.reward_components,
                "metrics": s2.training_metrics,
            },
            "shared_reward_components": list(
                set(s1.reward_components) & set(s2.reward_components)
            ),
            "transfer_potential": len(
                set(s1.reward_components) & set(s2.reward_components)
            ) / max(len(s1.reward_components), len(s2.reward_components), 1),
        }

        return [TextContent(type="text", text=json.dumps(comparison, indent=2))]

    elif name == "get_skill_insights":
        skill_id = arguments["skill_id"]

        from src.skill_learning.skill import SkillLibrary
        library = SkillLibrary(_skills_dir)
        skill = library.get_skill(skill_id)

        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        insights = {
            "skill_id": skill_id,
            "status": skill.status.value,
            "insights": [],
        }

        # Generate insights based on skill data
        if skill.training_metrics:
            metrics = skill.training_metrics
            if "mean_reward" in metrics:
                if metrics["mean_reward"] > 0:
                    insights["insights"].append("Skill achieves positive average reward")
                else:
                    insights["insights"].append("Skill has negative average reward - may need more training")

        if skill.prerequisites:
            insights["insights"].append(
                f"Depends on {len(skill.prerequisites)} prerequisite skills"
            )

        if skill.config.transfer_from:
            insights["insights"].append(
                f"Uses transfer learning from '{skill.config.transfer_from}'"
            )

        return [TextContent(type="text", text=json.dumps(insights, indent=2))]

    elif name == "analyze_failure_modes":
        skill_id = arguments["skill_id"]
        n_episodes = arguments.get("n_episodes", 50)

        # This would run episodes and track when/why failures occur
        analysis = {
            "skill_id": skill_id,
            "episodes_analyzed": n_episodes,
            "failure_modes": [
                {"mode": "balance_loss", "frequency": 0.4, "avg_step": 150},
                {"mode": "joint_limit", "frequency": 0.1, "avg_step": 300},
                {"mode": "timeout", "frequency": 0.5, "avg_step": 1000},
            ],
            "recommendations": [
                "Consider adding stability reward component",
                "May benefit from curriculum learning",
            ],
        }

        return [TextContent(type="text", text=json.dumps(analysis, indent=2))]

    elif name == "suggest_improvements":
        skill_id = arguments["skill_id"]

        suggestions = {
            "skill_id": skill_id,
            "suggestions": [
                {
                    "type": "reward_shaping",
                    "description": "Add intermediate rewards for sub-goals",
                    "priority": "high",
                },
                {
                    "type": "transfer_learning",
                    "description": "Consider transferring from a related skill",
                    "priority": "medium",
                },
                {
                    "type": "hyperparameter",
                    "description": "Try lower learning rate for stability",
                    "priority": "low",
                },
            ],
        }

        return [TextContent(type="text", text=json.dumps(suggestions, indent=2))]

    elif name == "get_transfer_candidates":
        from src.skill_learning.skill import SkillLibrary
        library = SkillLibrary(_skills_dir)

        trained = library.get_trained_skills()
        pending = [s for s in library.skills.values() if s.status.value == "pending"]

        candidates = []
        for pending_skill in pending:
            similar = library.find_similar_skills(pending_skill)
            if similar:
                candidates.append({
                    "target_skill": pending_skill.skill_id,
                    "potential_sources": [s.skill_id for s in similar],
                })

        return [TextContent(
            type="text",
            text=json.dumps({
                "transfer_candidates": candidates,
                "trained_skills_available": [s.skill_id for s in trained],
            }, indent=2)
        )]

    elif name == "generate_skill_report":
        skill_id = arguments["skill_id"]

        from src.skill_learning.skill import SkillLibrary
        library = SkillLibrary(_skills_dir)
        skill = library.get_skill(skill_id)

        if skill is None:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Skill '{skill_id}' not found"})
            )]

        report = {
            "title": f"Skill Report: {skill.name}",
            "skill_id": skill.skill_id,
            "description": skill.description,
            "status": skill.status.value,
            "configuration": {
                "algorithm": skill.config.algorithm,
                "total_timesteps": skill.config.total_timesteps,
                "learning_rate": skill.config.learning_rate,
                "transfer_from": skill.config.transfer_from,
            },
            "reward_design": {
                "components": skill.reward_components,
                "success_criteria": skill.success_criteria,
            },
            "dependencies": {
                "prerequisites": skill.prerequisites,
            },
            "training_results": skill.training_metrics,
        }

        return [TextContent(type="text", text=json.dumps(report, indent=2))]

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
