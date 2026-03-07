"""Robot specification module."""
from src.robot.robot_spec import (
    G1RobotSpec,
    get_robot_spec,
    get_reward_components_for_skill,
    get_skill_template,
    list_available_skills,
    REWARD_COMPONENTS,
    SKILL_TEMPLATES,
    JointGroup,
)

__all__ = [
    "G1RobotSpec",
    "get_robot_spec",
    "get_reward_components_for_skill",
    "get_skill_template",
    "list_available_skills",
    "REWARD_COMPONENTS",
    "SKILL_TEMPLATES",
    "JointGroup",
]
