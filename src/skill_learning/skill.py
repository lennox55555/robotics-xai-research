"""
Skill Definition and Management

A Skill represents a learnable behavior that can be:
- Trained via reinforcement learning
- Composed with other skills
- Used for transfer learning to new skills
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np


class SkillStatus(Enum):
    """Status of a skill in the learning pipeline."""
    PENDING = "pending"           # Not yet trained
    TRAINING = "training"         # Currently being trained
    TRAINED = "trained"           # Successfully trained
    FAILED = "failed"             # Training failed
    COMPOSED = "composed"         # Composed from other skills


@dataclass
class SkillConfig:
    """Configuration for training a skill."""
    algorithm: str = "PPO"
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99

    # Transfer learning settings
    transfer_from: Optional[str] = None  # Skill ID to transfer from
    transfer_layers: List[str] = field(default_factory=lambda: ["policy"])
    freeze_transferred: bool = False


@dataclass
class Skill:
    """
    Represents a single learnable skill for the robot.

    Examples:
        - "stand_upright": Maintain balance while standing
        - "walk_forward": Walk forward N steps
        - "jump": Perform a vertical jump
        - "backflip": Execute a backflip
    """

    # Identity
    skill_id: str                          # Unique identifier (e.g., "walk_forward")
    name: str                              # Human-readable name
    description: str                       # What this skill does

    # Task specification
    success_criteria: str                  # How to determine if skill succeeded
    reward_components: List[str]           # Components of the reward function
    termination_conditions: List[str]      # When to end an episode

    # Dependencies and composition
    prerequisites: List[str] = field(default_factory=list)  # Skills that must be learned first
    parent_skill: Optional[str] = None     # If this is a sub-skill of a larger task

    # Training configuration
    config: SkillConfig = field(default_factory=SkillConfig)

    # Status tracking
    status: SkillStatus = SkillStatus.PENDING
    training_metrics: Dict[str, Any] = field(default_factory=dict)

    # File paths (set after training)
    model_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['config'] = asdict(self.config)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Skill':
        """Create from dictionary."""
        data['status'] = SkillStatus(data['status'])
        data['config'] = SkillConfig(**data['config'])
        return cls(**data)

    def save(self, skills_dir: Path):
        """Save skill definition to JSON."""
        skill_path = skills_dir / "configs" / f"{self.skill_id}.json"
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skill_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, skill_id: str, skills_dir: Path) -> 'Skill':
        """Load skill definition from JSON."""
        skill_path = skills_dir / "configs" / f"{skill_id}.json"
        with open(skill_path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class TaskDecomposition:
    """
    Represents a complex task broken down into learnable skills.

    Example:
        Task: "Walk 10 steps and do a backflip"
        Skills: [
            Skill("balance"),
            Skill("walk_forward"),
            Skill("jump"),
            Skill("backflip")
        ]
        Execution order: balance -> walk_forward -> jump -> backflip
    """

    task_id: str
    original_prompt: str                   # The user's original request
    skills: List[Skill] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # Skill IDs in order

    # LLM reasoning
    llm_reasoning: str = ""                # LLM's explanation of the decomposition

    def add_skill(self, skill: Skill):
        """Add a skill to this task."""
        self.skills.append(skill)
        if skill.skill_id not in self.execution_order:
            self.execution_order.append(skill.skill_id)

    def get_next_skill_to_train(self) -> Optional[Skill]:
        """Get the next skill that needs training (respecting prerequisites)."""
        for skill_id in self.execution_order:
            skill = next((s for s in self.skills if s.skill_id == skill_id), None)
            if skill and skill.status == SkillStatus.PENDING:
                # Check if prerequisites are met
                prereqs_met = all(
                    any(s.skill_id == p and s.status == SkillStatus.TRAINED
                        for s in self.skills)
                    for p in skill.prerequisites
                )
                if prereqs_met:
                    return skill
        return None

    def all_skills_trained(self) -> bool:
        """Check if all skills are trained."""
        return all(s.status == SkillStatus.TRAINED for s in self.skills)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'original_prompt': self.original_prompt,
            'skills': [s.to_dict() for s in self.skills],
            'execution_order': self.execution_order,
            'llm_reasoning': self.llm_reasoning,
        }

    def save(self, tasks_dir: Path):
        """Save task decomposition."""
        task_path = tasks_dir / f"{self.task_id}.json"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SkillLibrary:
    """
    Manages all learned skills.

    Provides:
    - Skill lookup and retrieval
    - Finding similar skills for transfer learning
    - Skill composition
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self._load_all_skills()

    def _load_all_skills(self):
        """Load all skill definitions from disk."""
        configs_dir = self.skills_dir / "configs"
        if configs_dir.exists():
            for skill_file in configs_dir.glob("*.json"):
                skill = Skill.load(skill_file.stem, self.skills_dir)
                self.skills[skill.skill_id] = skill

    def add_skill(self, skill: Skill):
        """Add a new skill to the library."""
        self.skills[skill.skill_id] = skill
        skill.save(self.skills_dir)

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self.skills.get(skill_id)

    def get_trained_skills(self) -> List[Skill]:
        """Get all trained skills."""
        return [s for s in self.skills.values() if s.status == SkillStatus.TRAINED]

    def find_similar_skills(self, skill: Skill) -> List[Skill]:
        """
        Find skills that might be useful for transfer learning.

        Looks for skills with:
        - Similar reward components
        - Related success criteria
        - Trained status
        """
        similar = []
        trained = self.get_trained_skills()

        for trained_skill in trained:
            # Check for overlapping reward components
            overlap = set(skill.reward_components) & set(trained_skill.reward_components)
            if overlap:
                similar.append(trained_skill)

        return similar

    def get_skill_graph(self) -> Dict[str, List[str]]:
        """Get prerequisite graph for visualization."""
        return {
            skill_id: skill.prerequisites
            for skill_id, skill in self.skills.items()
        }
