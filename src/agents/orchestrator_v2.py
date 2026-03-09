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

        # Include ALL reward components
        reward_components = "\n".join([
            f"- **{name}**: {comp['description']} (for: {', '.join(comp['relevant_skills'])})"
            for name, comp in REWARD_COMPONENTS.items()
        ])

        return f"""You are an expert robotics AI trainer for the Unitree G1 humanoid robot.

{robot_context}

## Available Skill Templates (pre-defined)
{available_skills}

## ALL Available Reward Components
{reward_components}

## Your Capabilities
1. **Design Custom Skills**: Create skill definitions with any combination of reward components
2. **Set Reward Weights**: Configure how much each reward component contributes
3. **Start Training**: Begin RL training with specified timesteps
4. **Monitor Progress**: Track training metrics in real-time

## Tool Format
Use these tools to create and train skills:

### Define a skill with custom rewards and weights:
[TOOL: define_skill]
skill_id: raise_hand
name: Raise Right Hand
description: Raise the right hand above shoulder height while standing
reward_components: upright_reward, height_reward, com_stability, right_hand_height, energy_efficiency
reward_weights: upright_reward=1.0, height_reward=0.5, com_stability=0.5, right_hand_height=2.0, energy_efficiency=0.1
success_criteria: hand above 1.5m for 100 timesteps
timesteps: 500000
[/TOOL]

### Start training:
[TOOL: start_training]
skill_id: raise_hand
timesteps: 500000
[/TOOL]

### Check status:
[TOOL: check_status]
skill_id: raise_hand
[/TOOL]

## Key Body Parts for Skills
- **Hands**: right_hand_palm_link, left_hand_palm_link (for reaching, waving, grasping)
- **Feet**: left_ankle_roll_link, right_ankle_roll_link (for balance, walking)
- **Torso**: torso_link (for posture, balance)
- **Arms**: Use shoulder/elbow/wrist joints for arm movements

## Guidelines
- Always specify reward_weights to prioritize what the robot should optimize
- Higher weight = more important for the skill
- Include balance rewards (upright_reward, height_reward) for standing skills
- Use appropriate body-part rewards (right_hand_height for hand tasks)
- Be direct and confident - immediately create skills when asked
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
        """Define a new skill with custom rewards and weights."""
        skill_id = params.get("skill_id", "unnamed_skill")
        name = params.get("name", skill_id)
        description = params.get("description", "")
        reward_str = params.get("reward_components", "upright_reward")
        weights_str = params.get("reward_weights", "")
        success_criteria = params.get("success_criteria", "")
        timesteps = int(params.get("timesteps", 500_000))

        # Parse reward components
        reward_components = [r.strip() for r in reward_str.split(",")]

        # Parse reward weights (format: "comp1=1.0, comp2=2.0")
        reward_weights = {}
        if weights_str:
            for item in weights_str.split(","):
                if "=" in item:
                    comp, weight = item.strip().split("=")
                    reward_weights[comp.strip()] = float(weight.strip())

        # Fill in default weights for unspecified components
        for comp in reward_components:
            if comp not in reward_weights:
                # Use default from REWARD_COMPONENTS if available
                if comp in REWARD_COMPONENTS:
                    reward_weights[comp] = REWARD_COMPONENTS[comp].get("weight_default", 1.0)
                else:
                    reward_weights[comp] = 1.0

        # Check for template (but allow override with custom params)
        template = get_skill_template(skill_id)
        if template and not weights_str:
            # Use template values if no custom weights provided
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
            # Custom skill with specified rewards
            skill_def = {
                "skill_id": skill_id,
                "name": name,
                "description": description,
                "reward_components": reward_components,
                "reward_weights": reward_weights,
                "success_criteria": success_criteria,
                "termination_conditions": ["com_height < 0.4"],
                "training_config": {
                    "algorithm": "PPO",
                    "total_timesteps": timesteps,
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

        weights_summary = ", ".join([f"{k}={v}" for k, v in reward_weights.items()])
        return f"""Skill '{name}' defined successfully:
- Reward components: {', '.join(reward_components)}
- Reward weights: {weights_summary}
- Training timesteps: {skill_def['training_config']['total_timesteps']:,}
- Config saved to: {config_path}"""

    def _start_training(self, params: Dict) -> str:
        """Signal that training should start. Returns a special marker for the app to handle."""
        skill_id = params.get("skill_id", self.state.current_skill)
        timesteps = int(params.get("timesteps", 500_000))

        if not skill_id:
            return "No skill specified. Define a skill first."

        skill_def = self.state.skills_defined.get(skill_id)
        if not skill_def:
            # Try to load from config file
            config_path = self.skills_dir / "configs" / f"{skill_id}.json"
            if config_path.exists():
                with open(config_path) as f:
                    skill_def = json.load(f)
                self.state.skills_defined[skill_id] = skill_def
            else:
                # Try template
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

        self.state.current_skill = skill_id
        algorithm = skill_def.get("training_config", {}).get("algorithm", "PPO")
        lr = skill_def.get("training_config", {}).get("learning_rate", 3e-4)

        # Return a special marker that app.py will detect and use to start training
        return f"""[START_TRAINING:{skill_id}:{timesteps}]

Training **{skill_def['name']}** is starting!

- Algorithm: {algorithm}
- Timesteps: {timesteps:,}
- Learning rate: {lr}

Watch the simulation to see the robot learn in real-time."""

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
