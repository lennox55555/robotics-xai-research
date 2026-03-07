"""
LLM Orchestrator for Skill Learning

The orchestrator uses an LLM to:
1. Understand the robot's capabilities (joints, actuators)
2. Break down complex tasks into learnable skills
3. Decide when to train new models vs reuse existing skills
4. Recommend transfer learning opportunities
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.skill_learning.skill import (
    Skill, SkillConfig, SkillStatus, TaskDecomposition, SkillLibrary
)


SYSTEM_PROMPT = """You are an AI orchestrator for a humanoid robot learning system.

## Your Role
You help break down complex movement tasks into learnable skills that can be trained via reinforcement learning.

## Robot Specification
{robot_spec}

## Available Trained Skills
{available_skills}

## Your Capabilities
1. **Task Decomposition**: Break complex tasks into smaller, learnable skills
2. **Skill Design**: Define reward functions, success criteria, and termination conditions
3. **Transfer Learning**: Identify which existing skills can help learn new skills faster
4. **Skill Composition**: Combine multiple skills to achieve complex behaviors

## Output Format
When decomposing a task, respond with valid JSON in this format:
```json
{{
    "reasoning": "Your step-by-step reasoning about how to decompose this task",
    "skills": [
        {{
            "skill_id": "unique_snake_case_id",
            "name": "Human Readable Name",
            "description": "What this skill does",
            "success_criteria": "How to measure success",
            "reward_components": ["component1", "component2"],
            "termination_conditions": ["condition1", "condition2"],
            "prerequisites": ["skill_id_of_prerequisite"],
            "transfer_from": "skill_id_to_transfer_from_or_null",
            "estimated_difficulty": "easy|medium|hard"
        }}
    ],
    "execution_order": ["skill_id_1", "skill_id_2"]
}}
```

## Reward Component Options
Common reward components you can specify:
- "height_reward": Reward for maintaining/achieving height
- "upright_reward": Reward for staying upright (torso orientation)
- "velocity_forward": Reward for forward movement
- "velocity_target": Reward for matching target velocity
- "energy_efficiency": Penalty for excessive control effort
- "stability": Reward for low joint velocities
- "foot_contact": Reward for proper foot contact patterns
- "joint_limits": Penalty for approaching joint limits
- "symmetry": Reward for symmetric movements
- "smoothness": Penalty for jerky movements

## Important Guidelines
1. Start with foundational skills (balance, stand) before complex ones (walk, jump)
2. Each skill should be learnable in isolation
3. Suggest transfer learning when a new skill is similar to an existing one
4. Keep skills focused - one main objective per skill
5. Consider the physical constraints of the robot
"""


@dataclass
class RobotSpecification:
    """Specification of the robot's physical capabilities."""
    name: str
    num_joints: int
    num_actuators: int
    joint_names: List[str]
    description: str

    def to_prompt(self) -> str:
        return f"""
Robot: {self.name}
- Total joints: {self.num_joints}
- Total actuators: {self.num_actuators}
- Joint groups:
{self._format_joints()}
- Description: {self.description}
"""

    def _format_joints(self) -> str:
        lines = []
        for name in self.joint_names[:20]:  # Limit for prompt
            lines.append(f"  - {name}")
        if len(self.joint_names) > 20:
            lines.append(f"  ... and {len(self.joint_names) - 20} more")
        return "\n".join(lines)


class LLMOrchestrator:
    """
    Orchestrates the skill learning process using an LLM.

    The orchestrator:
    1. Takes high-level task descriptions from users
    2. Decomposes them into learnable skills
    3. Manages the training pipeline
    4. Tracks progress and suggests next steps
    """

    def __init__(
        self,
        robot_spec: RobotSpecification,
        skill_library: SkillLibrary,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.robot_spec = robot_spec
        self.skill_library = skill_library
        self.model = model
        self.client = anthropic.Anthropic()

        self.tasks: Dict[str, TaskDecomposition] = {}

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current context."""
        trained_skills = self.skill_library.get_trained_skills()

        if trained_skills:
            skills_text = "\n".join([
                f"- {s.skill_id}: {s.description}"
                for s in trained_skills
            ])
        else:
            skills_text = "No skills trained yet. Start with foundational skills."

        return SYSTEM_PROMPT.format(
            robot_spec=self.robot_spec.to_prompt(),
            available_skills=skills_text,
        )

    def decompose_task(self, task_prompt: str) -> TaskDecomposition:
        """
        Use LLM to decompose a complex task into learnable skills.

        Args:
            task_prompt: Natural language description of the task
                         e.g., "Walk 10 steps forward and do a backflip"

        Returns:
            TaskDecomposition with skills to train
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"""Please decompose this task into learnable skills:

Task: {task_prompt}

Remember to:
1. Start with foundational skills if they don't exist
2. Consider transfer learning opportunities
3. Define clear reward components for each skill
4. Output valid JSON only
"""
                }
            ]
        )

        # Parse the response
        response_text = message.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        data = json.loads(json_str.strip())

        # Create TaskDecomposition
        task_id = f"task_{len(self.tasks) + 1}"
        decomposition = TaskDecomposition(
            task_id=task_id,
            original_prompt=task_prompt,
            llm_reasoning=data.get("reasoning", ""),
        )

        # Create skills
        for skill_data in data["skills"]:
            config = SkillConfig(
                transfer_from=skill_data.get("transfer_from"),
            )

            skill = Skill(
                skill_id=skill_data["skill_id"],
                name=skill_data["name"],
                description=skill_data["description"],
                success_criteria=skill_data["success_criteria"],
                reward_components=skill_data["reward_components"],
                termination_conditions=skill_data["termination_conditions"],
                prerequisites=skill_data.get("prerequisites", []),
                config=config,
            )

            decomposition.add_skill(skill)
            self.skill_library.add_skill(skill)

        decomposition.execution_order = data["execution_order"]
        self.tasks[task_id] = decomposition

        return decomposition

    def get_next_skill_to_train(self, task_id: str) -> Optional[Skill]:
        """Get the next skill that should be trained for a task."""
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].get_next_skill_to_train()

    def suggest_transfer_learning(self, skill: Skill) -> Optional[str]:
        """
        Ask LLM to suggest which existing skill to transfer from.

        Returns skill_id of the best skill to transfer from, or None.
        """
        trained = self.skill_library.get_trained_skills()
        if not trained:
            return None

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"""I need to train this new skill:

Skill: {skill.name}
Description: {skill.description}
Reward components: {skill.reward_components}

Which existing trained skill would be best to transfer from?
Respond with just the skill_id, or "none" if no good transfer candidate exists.

Available trained skills:
{json.dumps([{'skill_id': s.skill_id, 'description': s.description, 'reward_components': s.reward_components} for s in trained], indent=2)}
"""
                }
            ]
        )

        response = message.content[0].text.strip().lower()
        if response == "none":
            return None

        # Validate it's an actual skill
        if any(s.skill_id == response for s in trained):
            return response

        return None

    def explain_decision(self, skill: Skill) -> str:
        """Get LLM to explain why it designed a skill this way."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"""Explain why you designed this skill with these specific components:

Skill: {skill.name}
Description: {skill.description}
Reward components: {skill.reward_components}
Success criteria: {skill.success_criteria}
Prerequisites: {skill.prerequisites}

Explain:
1. Why these reward components?
2. Why these prerequisites?
3. What makes this skill learnable?
4. How will it compose with other skills?
"""
                }
            ]
        )

        return message.content[0].text

    def review_training_progress(
        self,
        skill: Skill,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ask LLM to review training progress and suggest adjustments.

        Returns suggestions for:
        - Continue training
        - Adjust hyperparameters
        - Modify reward function
        - Start over with different approach
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"""Review the training progress for this skill:

Skill: {skill.name}
Description: {skill.description}
Reward components: {skill.reward_components}

Training metrics:
{json.dumps(metrics, indent=2)}

Please analyze:
1. Is training progressing well?
2. Should we adjust hyperparameters?
3. Should we modify the reward function?
4. Any other suggestions?

Respond with JSON:
{{
    "assessment": "good|struggling|failed",
    "continue_training": true/false,
    "suggestions": ["suggestion1", "suggestion2"],
    "reward_adjustments": {{"component": "adjustment"}},
    "reasoning": "Your analysis"
}}
"""
                }
            ]
        )

        response_text = message.content[0].text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        else:
            json_str = response_text

        return json.loads(json_str.strip())


def create_g1_robot_spec() -> RobotSpecification:
    """Create robot specification for Unitree G1."""
    joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        "waist_yaw", "waist_roll", "waist_pitch",
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "left_thumb_0", "left_thumb_1", "left_thumb_2",
        "left_index_0", "left_index_1", "left_middle_0", "left_middle_1",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        "right_thumb_0", "right_thumb_1", "right_thumb_2",
        "right_index_0", "right_index_1", "right_middle_0", "right_middle_1",
    ]

    return RobotSpecification(
        name="Unitree G1 Humanoid with Dexterous Hands",
        num_joints=44,
        num_actuators=43,
        joint_names=joint_names,
        description="""
A full humanoid robot with:
- Bipedal locomotion (12 DOF legs)
- Articulated torso (3 DOF waist)
- Dual arms with 7 DOF each
- Dexterous hands with thumb, index, and middle fingers
Capable of walking, running, jumping, and manipulation tasks.
"""
    )
