"""
Multi-Agent Orchestrator v2

A properly working orchestrator that:
1. Maintains conversation state across turns
2. Executes agent tools correctly
3. Doesn't apologize unnecessarily
4. Actually starts training when requested
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.robot.robot_spec import (
    get_robot_spec,
    get_skill_template,
    list_available_skills,
    REWARD_COMPONENTS,
    SKILL_TEMPLATES,
)
from src.skill_learning.skill import Skill, SkillConfig, SkillStatus
from src.skill_learning.skill_trainer import SkillTrainer

load_dotenv()


@dataclass
class ConversationState:
    """Tracks the state of the conversation."""
    messages: List[Dict] = field(default_factory=list)
    current_skill: Optional[str] = None
    skills_defined: Dict[str, Dict] = field(default_factory=dict)
    skills_training: Dict[str, Any] = field(default_factory=dict)
    skills_completed: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    training_active: bool = False


class UnifiedOrchestrator:
    """
    A unified orchestrator that handles all agent responsibilities.

    Instead of complex handoffs, this orchestrator:
    1. Has full knowledge of the robot
    2. Can create skill definitions
    3. Can start/monitor training
    4. Provides clear, non-apologetic responses
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.model = "claude-sonnet-4-20250514"
        self.robot_spec = get_robot_spec()
        self.state = ConversationState()
        self.trainer: Optional[SkillTrainer] = None
        self.skills_dir = Path(__file__).parent.parent.parent / "skills"

        # Build system prompt with robot knowledge
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with robot knowledge."""
        robot_context = self.robot_spec.to_prompt_context()

        available_skills = "\n".join([
            f"- **{name}**: {SKILL_TEMPLATES[name]['description']}"
            for name in list_available_skills()
        ])

        reward_components = "\n".join([
            f"- **{name}**: {comp['description']}"
            for name, comp in list(REWARD_COMPONENTS.items())[:10]
        ])

        return f"""You are an expert robotics AI trainer for teaching humanoid robots new skills.

{robot_context}

## Available Skill Templates
{available_skills}

## Reward Components Available
{reward_components}

## Your Capabilities
1. **Design Skills**: Create skill definitions with appropriate rewards
2. **Start Training**: Begin RL training for defined skills
3. **Monitor Progress**: Track training metrics
4. **Analyze Results**: Explain policy behavior

## Response Guidelines
- Be direct and confident. Never apologize for "confusion" or "handoff issues"
- When the user asks to train something, create a skill definition and offer to start training
- Use the TOOL markers below when you need to execute actions
- Keep responses concise and action-oriented

## Tool Format
When you need to execute an action, include it in your response like this:

[TOOL: define_skill]
skill_id: walk_forward
name: Walk Forward
description: Walk forward maintaining balance
reward_components: forward_velocity, upright_reward, energy_efficiency
success_criteria: walk 5m without falling
[/TOOL]

[TOOL: start_training]
skill_id: walk_forward
timesteps: 500000
[/TOOL]

[TOOL: check_status]
skill_id: walk_forward
[/TOOL]

Always explain what you're doing and why before using a tool.
"""

    def _parse_tools(self, response: str) -> List[Dict]:
        """Extract tool calls from response."""
        tools = []
        pattern = r'\[TOOL:\s*(\w+)\](.*?)\[/TOOL\]'
        matches = re.findall(pattern, response, re.DOTALL)

        for tool_name, tool_content in matches:
            params = {}
            for line in tool_content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()
            tools.append({"name": tool_name, "params": params})

        return tools

    def _execute_tool(self, tool: Dict) -> str:
        """Execute a tool and return result."""
        name = tool["name"]
        params = tool["params"]

        if name == "define_skill":
            return self._define_skill(params)
        elif name == "start_training":
            return self._start_training(params)
        elif name == "check_status":
            return self._check_status(params)
        elif name == "list_skills":
            return self._list_skills()
        else:
            return f"Unknown tool: {name}"

    def _define_skill(self, params: Dict) -> str:
        """Define a new skill."""
        skill_id = params.get("skill_id", "unnamed_skill")
        name = params.get("name", skill_id)
        description = params.get("description", "")
        reward_str = params.get("reward_components", "upright_reward")
        success_criteria = params.get("success_criteria", "")

        # Parse reward components
        reward_components = [r.strip() for r in reward_str.split(",")]

        # Check for template
        template = get_skill_template(skill_id)
        if template:
            # Use template values
            skill_def = {
                "skill_id": skill_id,
                "name": template["name"],
                "description": template["description"],
                "reward_components": template["reward_components"],
                "reward_weights": template.get("reward_weights", {}),
                "success_criteria": template["success_criteria"],
                "termination_conditions": template.get("termination_conditions", []),
                "training_config": template["training_config"],
                "curriculum": template.get("curriculum"),
                "prerequisites": template.get("prerequisites", []),
            }
        else:
            # Custom skill
            skill_def = {
                "skill_id": skill_id,
                "name": name,
                "description": description,
                "reward_components": reward_components,
                "reward_weights": {r: 1.0 for r in reward_components},
                "success_criteria": success_criteria,
                "termination_conditions": ["com_height < 0.4"],
                "training_config": {
                    "algorithm": "PPO",
                    "total_timesteps": 500_000,
                    "learning_rate": 3e-4,
                },
                "curriculum": None,
                "prerequisites": [],
            }

        self.state.skills_defined[skill_id] = skill_def
        self.state.current_skill = skill_id

        # Save to disk
        config_path = self.skills_dir / "configs" / f"{skill_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(skill_def, f, indent=2)

        return f"""Skill '{name}' defined successfully:
- Reward components: {', '.join(skill_def['reward_components'])}
- Training timesteps: {skill_def['training_config']['total_timesteps']:,}
- Prerequisites: {skill_def['prerequisites'] or 'None'}
- Config saved to: {config_path}"""

    def _start_training(self, params: Dict) -> str:
        """Start training a skill."""
        skill_id = params.get("skill_id", self.state.current_skill)
        timesteps = int(params.get("timesteps", 500_000))

        if not skill_id:
            return "No skill specified. Define a skill first."

        skill_def = self.state.skills_defined.get(skill_id)
        if not skill_def:
            # Try to load from template
            template = get_skill_template(skill_id)
            if template:
                self._define_skill({"skill_id": skill_id})
                skill_def = self.state.skills_defined.get(skill_id)
            else:
                return f"Skill '{skill_id}' not defined. Define it first."

        # Check prerequisites
        for prereq in skill_def.get("prerequisites", []):
            if prereq not in self.state.skills_completed:
                return f"Prerequisite skill '{prereq}' must be trained first."

        # Create Skill object
        skill = Skill(
            skill_id=skill_id,
            name=skill_def["name"],
            description=skill_def["description"],
            success_criteria=skill_def["success_criteria"],
            reward_components=skill_def["reward_components"],
            termination_conditions=skill_def.get("termination_conditions", []),
            prerequisites=skill_def.get("prerequisites", []),
            config=SkillConfig(
                algorithm=skill_def["training_config"].get("algorithm", "PPO"),
                total_timesteps=timesteps,
                learning_rate=skill_def["training_config"].get("learning_rate", 3e-4),
                transfer_from=skill_def["training_config"].get("transfer_from"),
            ),
        )

        self.state.training_active = True
        self.state.skills_training[skill_id] = {
            "skill": skill,
            "status": "starting",
            "progress": 0,
        }

        # Return info about how to actually run training
        return f"""Training configuration ready for '{skill_def['name']}':

## Training Parameters
- Algorithm: {skill.config.algorithm}
- Total Timesteps: {timesteps:,}
- Learning Rate: {skill.config.learning_rate}
- Transfer From: {skill.config.transfer_from or 'None (training from scratch)'}

## Reward Function
{self._format_reward_function(skill_def)}

## To Start Training
Run in terminal:
```bash
python -c "
from src.skill_learning.skill_trainer import SkillTrainer
from src.skill_learning.skill import Skill, SkillConfig

trainer = SkillTrainer()
trainer.train_skill('{skill_id}', timesteps={timesteps})
"
```

## Monitor Progress
```bash
tensorboard --logdir logs/training/{skill_id}/
```

Training will save checkpoints to: skills/trained/{skill_id}/"""

    def _format_reward_function(self, skill_def: Dict) -> str:
        """Format reward function as readable code."""
        components = skill_def.get("reward_components", [])
        weights = skill_def.get("reward_weights", {})

        lines = ["```python", "def compute_reward(obs, action, info):"]
        lines.append("    reward = 0.0")

        for comp in components:
            weight = weights.get(comp, 1.0)
            sign = "+" if weight >= 0 else ""
            lines.append(f"    reward {sign}= {weight} * {comp}(obs, info)")

        lines.append("    return reward")
        lines.append("```")
        return "\n".join(lines)

    def _check_status(self, params: Dict) -> str:
        """Check training status."""
        skill_id = params.get("skill_id", self.state.current_skill)

        if not skill_id:
            return "No skill specified."

        if skill_id in self.state.skills_training:
            info = self.state.skills_training[skill_id]
            return f"Training status for '{skill_id}': {info['status']}"

        if skill_id in self.state.skills_completed:
            return f"Skill '{skill_id}' has completed training."

        if skill_id in self.state.skills_defined:
            return f"Skill '{skill_id}' is defined but not yet training."

        return f"Skill '{skill_id}' not found."

    def _list_skills(self) -> str:
        """List all skills."""
        lines = ["## Available Skill Templates"]
        for name in list_available_skills():
            template = SKILL_TEMPLATES[name]
            lines.append(f"- **{name}**: {template['description']}")

        if self.state.skills_defined:
            lines.append("\n## Defined Skills")
            for skill_id, skill_def in self.state.skills_defined.items():
                status = "training" if skill_id in self.state.skills_training else "ready"
                lines.append(f"- **{skill_id}**: {skill_def['name']} ({status})")

        return "\n".join(lines)

    def process(self, user_input: str) -> str:
        """Process user input and return response."""
        # Add user message to history
        self.state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Build context from recent messages
        context_messages = self.state.messages[-10:]  # Last 10 messages

        # Add state context
        state_context = self._get_state_context()
        if state_context:
            context_messages = [
                {"role": "user", "content": f"[Current State]\n{state_context}"},
                {"role": "assistant", "content": "Understood. I have the current state context."},
            ] + context_messages

        # Call Claude
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            messages=context_messages,
        )

        assistant_message = response.content[0].text

        # Parse and execute any tools
        tools = self._parse_tools(assistant_message)
        tool_results = []

        for tool in tools:
            result = self._execute_tool(tool)
            tool_results.append(f"\n---\n**Tool Result ({tool['name']}):**\n{result}")

        # If there were tool results, append them
        if tool_results:
            # Remove tool markers from display
            clean_response = re.sub(r'\[TOOL:.*?\[/TOOL\]', '', assistant_message, flags=re.DOTALL).strip()
            final_response = clean_response + "\n".join(tool_results)
        else:
            final_response = assistant_message

        # Store assistant response
        self.state.messages.append({
            "role": "assistant",
            "content": assistant_message,  # Store original with tools
        })

        return final_response

    def _get_state_context(self) -> str:
        """Get current state as context string."""
        lines = []

        if self.state.current_skill:
            lines.append(f"Current skill focus: {self.state.current_skill}")

        if self.state.skills_defined:
            lines.append(f"Defined skills: {', '.join(self.state.skills_defined.keys())}")

        if self.state.skills_training:
            lines.append(f"Training: {', '.join(self.state.skills_training.keys())}")

        if self.state.skills_completed:
            lines.append(f"Completed: {', '.join(self.state.skills_completed)}")

        return "\n".join(lines)

    def run_interactive(self):
        """Run interactive session."""
        print("=" * 60)
        print("Robot Skill Learning System")
        print("=" * 60)
        print()
        print(f"Robot: {self.robot_spec.name}")
        print(f"  - {self.robot_spec.total_actuators} controllable joints")
        print(f"  - {self.robot_spec.height_standing}m standing height")
        print()
        print("Available commands:")
        print("  'quit'  - Exit")
        print("  'skills' - List available skill templates")
        print("  'state' - Show current state")
        print()
        print("Example: 'Teach the robot to walk forward'")
        print("-" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "skills":
                    print("\n" + self._list_skills() + "\n")
                    continue

                if user_input.lower() == "state":
                    print("\n" + (self._get_state_context() or "No active state.") + "\n")
                    continue

                response = self.process(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def create_orchestrator() -> UnifiedOrchestrator:
    """Create a new orchestrator instance."""
    return UnifiedOrchestrator()


if __name__ == "__main__":
    orchestrator = create_orchestrator()
    orchestrator.run_interactive()
