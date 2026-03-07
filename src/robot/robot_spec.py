"""
Robot Specification Module

Provides detailed knowledge about the Unitree G1 humanoid robot:
- Joint names, types, and limits
- Body groups (legs, arms, hands, torso)
- Actuator mappings
- Reward component definitions
- Pre-defined skill templates
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class JointGroup(Enum):
    """Logical groupings of joints."""
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"


@dataclass
class JointSpec:
    """Specification for a single joint."""
    name: str
    index: int
    group: JointGroup
    joint_type: str  # "hinge", "free", "ball", "slide"
    range_min: float = -3.14
    range_max: float = 3.14
    max_torque: float = 100.0
    has_actuator: bool = True


@dataclass
class G1RobotSpec:
    """
    Complete specification for the Unitree G1 Humanoid robot.

    This provides all the information needed for:
    - Designing reward functions
    - Setting up training experiments
    - Understanding robot capabilities
    """

    # Basic specs
    name: str = "Unitree G1"
    total_dof: int = 49  # Including floating base (6 DOF)
    total_joints: int = 44
    total_actuators: int = 43
    total_bodies: int = 45

    # Physical properties
    height_standing: float = 1.32  # meters
    mass: float = 35.0  # kg (approximate)

    # XML path
    xml_path: str = "mujoco_menagerie/unitree_g1/g1_with_hands.xml"

    # Joint groups
    joint_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "left_leg": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        "right_leg": [
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        "torso": [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        ],
        "left_arm": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        "right_arm": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "left_hand": [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
        ],
        "right_hand": [
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
        ],
    })

    # Actuator indices for control
    actuator_groups: Dict[str, List[int]] = field(default_factory=lambda: {
        "left_leg": [0, 1, 2, 3, 4, 5],
        "right_leg": [6, 7, 8, 9, 10, 11],
        "torso": [12, 13, 14],
        "left_arm": [15, 16, 17, 18, 19, 20, 21],
        "right_arm": [29, 30, 31, 32, 33, 34, 35],
        "left_hand": [22, 23, 24, 25, 26, 27, 28],
        "right_hand": [36, 37, 38, 39, 40, 41, 42],
    })

    def get_joints_for_skill(self, skill_type: str) -> List[str]:
        """Get relevant joints for a given skill type."""
        skill_joint_map = {
            "balance": self.joint_groups["left_leg"] + self.joint_groups["right_leg"] + self.joint_groups["torso"],
            "walk": self.joint_groups["left_leg"] + self.joint_groups["right_leg"] + self.joint_groups["torso"],
            "run": self.joint_groups["left_leg"] + self.joint_groups["right_leg"] + self.joint_groups["torso"] + self.joint_groups["left_arm"] + self.joint_groups["right_arm"],
            "jump": self.joint_groups["left_leg"] + self.joint_groups["right_leg"] + self.joint_groups["torso"],
            "grasp": self.joint_groups["left_hand"] + self.joint_groups["right_hand"] + self.joint_groups["left_arm"] + self.joint_groups["right_arm"],
            "wave": self.joint_groups["right_arm"],
            "full_body": sum(self.joint_groups.values(), []),
        }
        return skill_joint_map.get(skill_type, skill_joint_map["full_body"])

    def get_actuator_indices_for_skill(self, skill_type: str) -> List[int]:
        """Get relevant actuator indices for a given skill type."""
        skill_actuator_map = {
            "balance": self.actuator_groups["left_leg"] + self.actuator_groups["right_leg"] + self.actuator_groups["torso"],
            "walk": self.actuator_groups["left_leg"] + self.actuator_groups["right_leg"] + self.actuator_groups["torso"],
            "run": self.actuator_groups["left_leg"] + self.actuator_groups["right_leg"] + self.actuator_groups["torso"] + self.actuator_groups["left_arm"] + self.actuator_groups["right_arm"],
            "jump": self.actuator_groups["left_leg"] + self.actuator_groups["right_leg"] + self.actuator_groups["torso"],
            "grasp": self.actuator_groups["left_hand"] + self.actuator_groups["right_hand"] + self.actuator_groups["left_arm"] + self.actuator_groups["right_arm"],
        }
        return skill_actuator_map.get(skill_type, list(range(self.total_actuators)))

    def to_prompt_context(self) -> str:
        """Generate a context string for LLM prompts."""
        return f"""## Robot Specification: {self.name}

### Physical Properties
- Height (standing): {self.height_standing}m
- Mass: {self.mass}kg
- Total DOF: {self.total_dof} (including 6-DOF floating base)
- Controllable joints: {self.total_actuators}

### Joint Groups
**Legs (12 joints each side)**:
- Hip: pitch, roll, yaw (3 DOF per leg)
- Knee: pitch (1 DOF per leg)
- Ankle: pitch, roll (2 DOF per leg)

**Torso (3 joints)**:
- Waist: yaw, roll, pitch

**Arms (7 joints each side)**:
- Shoulder: pitch, roll, yaw (3 DOF)
- Elbow: pitch (1 DOF)
- Wrist: roll, pitch, yaw (3 DOF)

**Hands (7 joints each side)**:
- Thumb: 3 joints
- Index finger: 2 joints
- Middle finger: 2 joints

### Key Joint Names
Left Leg: {', '.join(self.joint_groups['left_leg'])}
Right Leg: {', '.join(self.joint_groups['right_leg'])}
Torso: {', '.join(self.joint_groups['torso'])}

### Skill-Relevant Joint Groups
- Balance/Walking: legs (12) + torso (3) = 15 active joints
- Manipulation: arms (14) + hands (14) = 28 active joints
- Full body: all 43 actuated joints
"""


# Pre-defined reward components for the G1 robot
REWARD_COMPONENTS = {
    # Balance rewards
    "upright_reward": {
        "description": "Reward for maintaining upright torso orientation",
        "formula": "1.0 - abs(torso_pitch) - abs(torso_roll)",
        "weight_default": 1.0,
        "relevant_skills": ["balance", "walk", "run", "jump"],
    },
    "height_reward": {
        "description": "Reward for maintaining target standing height",
        "formula": "exp(-10 * (com_height - target_height)^2)",
        "weight_default": 0.5,
        "relevant_skills": ["balance", "walk", "stand"],
    },
    "com_stability": {
        "description": "Penalize center of mass velocity when it should be stable",
        "formula": "-norm(com_velocity)",
        "weight_default": 0.3,
        "relevant_skills": ["balance", "stand"],
    },

    # Locomotion rewards
    "forward_velocity": {
        "description": "Reward for moving forward at target speed",
        "formula": "exp(-2 * (forward_vel - target_vel)^2)",
        "weight_default": 2.0,
        "relevant_skills": ["walk", "run"],
    },
    "lateral_stability": {
        "description": "Penalize lateral (sideways) movement",
        "formula": "-abs(lateral_velocity)",
        "weight_default": 0.5,
        "relevant_skills": ["walk", "run"],
    },
    "gait_symmetry": {
        "description": "Reward symmetric leg movements",
        "formula": "-abs(left_leg_phase - right_leg_phase - 0.5)",
        "weight_default": 0.3,
        "relevant_skills": ["walk", "run"],
    },
    "foot_clearance": {
        "description": "Reward lifting feet during swing phase",
        "formula": "max(0, swing_foot_height - threshold)",
        "weight_default": 0.2,
        "relevant_skills": ["walk", "run"],
    },

    # Efficiency rewards
    "energy_efficiency": {
        "description": "Penalize excessive torque usage",
        "formula": "-sum(torques^2) / max_torque_sq",
        "weight_default": 0.1,
        "relevant_skills": ["all"],
    },
    "smoothness": {
        "description": "Penalize jerky movements (action changes)",
        "formula": "-sum((action - prev_action)^2)",
        "weight_default": 0.1,
        "relevant_skills": ["all"],
    },
    "joint_limit_penalty": {
        "description": "Penalize approaching joint limits",
        "formula": "-sum(max(0, |joint| - 0.9*limit)^2)",
        "weight_default": 0.5,
        "relevant_skills": ["all"],
    },

    # Safety rewards
    "fall_penalty": {
        "description": "Large penalty for falling",
        "formula": "-100 if com_height < fall_threshold else 0",
        "weight_default": 1.0,
        "relevant_skills": ["all"],
    },
    "self_collision_penalty": {
        "description": "Penalize self-collision",
        "formula": "-10 * num_self_collisions",
        "weight_default": 1.0,
        "relevant_skills": ["all"],
    },

    # Jump-specific
    "jump_height": {
        "description": "Reward for achieving jump height",
        "formula": "max(0, com_height - standing_height)",
        "weight_default": 5.0,
        "relevant_skills": ["jump"],
    },
    "landing_stability": {
        "description": "Reward stable landing after jump",
        "formula": "1.0 if stable_after_landing else 0",
        "weight_default": 2.0,
        "relevant_skills": ["jump"],
    },
}


# Pre-defined skill templates
SKILL_TEMPLATES = {
    "balance_stand": {
        "name": "Balance Standing",
        "description": "Maintain upright standing position without falling",
        "reward_components": ["upright_reward", "height_reward", "com_stability", "energy_efficiency", "fall_penalty"],
        "reward_weights": {"upright_reward": 2.0, "height_reward": 1.0, "com_stability": 0.5, "energy_efficiency": 0.1, "fall_penalty": 1.0},
        "success_criteria": "maintain standing for 500 timesteps with torso angle < 0.2 rad",
        "termination_conditions": ["com_height < 0.5", "torso_angle > 1.0"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 200_000,
            "learning_rate": 3e-4,
        },
        "curriculum": None,
        "prerequisites": [],
    },
    "walk_forward": {
        "name": "Walk Forward",
        "description": "Walk forward at moderate speed while maintaining balance",
        "reward_components": ["forward_velocity", "upright_reward", "gait_symmetry", "energy_efficiency", "fall_penalty", "lateral_stability"],
        "reward_weights": {"forward_velocity": 2.0, "upright_reward": 1.0, "gait_symmetry": 0.5, "energy_efficiency": 0.1, "fall_penalty": 1.0, "lateral_stability": 0.3},
        "success_criteria": "walk forward 5m without falling, maintaining velocity > 0.5 m/s",
        "termination_conditions": ["com_height < 0.5", "episode_length > 1000"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 1_000_000,
            "learning_rate": 1e-4,
            "transfer_from": "balance_stand",
        },
        "curriculum": [
            {"stage": 1, "target_velocity": 0.2, "max_steps": 200_000},
            {"stage": 2, "target_velocity": 0.5, "max_steps": 400_000},
            {"stage": 3, "target_velocity": 0.8, "max_steps": 400_000},
        ],
        "prerequisites": ["balance_stand"],
    },
    "walk_backward": {
        "name": "Walk Backward",
        "description": "Walk backward while maintaining balance",
        "reward_components": ["forward_velocity", "upright_reward", "gait_symmetry", "energy_efficiency", "fall_penalty"],
        "reward_weights": {"forward_velocity": -2.0, "upright_reward": 1.0, "gait_symmetry": 0.5, "energy_efficiency": 0.1, "fall_penalty": 1.0},
        "success_criteria": "walk backward 3m without falling",
        "termination_conditions": ["com_height < 0.5", "episode_length > 1000"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 500_000,
            "learning_rate": 1e-4,
            "transfer_from": "walk_forward",
        },
        "prerequisites": ["walk_forward"],
    },
    "turn_left": {
        "name": "Turn Left",
        "description": "Turn left while maintaining balance",
        "reward_components": ["upright_reward", "energy_efficiency", "fall_penalty"],
        "success_criteria": "complete 90 degree turn without falling",
        "termination_conditions": ["com_height < 0.5"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 300_000,
            "transfer_from": "balance_stand",
        },
        "prerequisites": ["balance_stand"],
    },
    "turn_right": {
        "name": "Turn Right",
        "description": "Turn right while maintaining balance",
        "reward_components": ["upright_reward", "energy_efficiency", "fall_penalty"],
        "success_criteria": "complete 90 degree turn without falling",
        "termination_conditions": ["com_height < 0.5"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 300_000,
            "transfer_from": "balance_stand",
        },
        "prerequisites": ["balance_stand"],
    },
    "jump": {
        "name": "Vertical Jump",
        "description": "Perform a vertical jump and land stably",
        "reward_components": ["jump_height", "landing_stability", "upright_reward", "fall_penalty"],
        "reward_weights": {"jump_height": 5.0, "landing_stability": 3.0, "upright_reward": 1.0, "fall_penalty": 1.0},
        "success_criteria": "achieve jump height > 0.1m and land without falling",
        "termination_conditions": ["com_height < 0.3", "episode_length > 500"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 2_000_000,
            "learning_rate": 5e-5,
            "transfer_from": "balance_stand",
        },
        "curriculum": [
            {"stage": 1, "target_height": 0.05, "max_steps": 500_000},
            {"stage": 2, "target_height": 0.1, "max_steps": 750_000},
            {"stage": 3, "target_height": 0.15, "max_steps": 750_000},
        ],
        "prerequisites": ["balance_stand"],
    },
    "run_forward": {
        "name": "Run Forward",
        "description": "Run forward at high speed",
        "reward_components": ["forward_velocity", "upright_reward", "gait_symmetry", "foot_clearance", "energy_efficiency", "fall_penalty"],
        "reward_weights": {"forward_velocity": 3.0, "upright_reward": 1.0, "gait_symmetry": 0.3, "foot_clearance": 0.3, "energy_efficiency": 0.05, "fall_penalty": 1.0},
        "success_criteria": "run forward at > 1.5 m/s for 3 seconds",
        "termination_conditions": ["com_height < 0.4", "episode_length > 500"],
        "training_config": {
            "algorithm": "PPO",
            "total_timesteps": 3_000_000,
            "learning_rate": 5e-5,
            "transfer_from": "walk_forward",
        },
        "curriculum": [
            {"stage": 1, "target_velocity": 1.0, "max_steps": 1_000_000},
            {"stage": 2, "target_velocity": 1.5, "max_steps": 1_000_000},
            {"stage": 3, "target_velocity": 2.0, "max_steps": 1_000_000},
        ],
        "prerequisites": ["walk_forward"],
    },
}


def get_robot_spec() -> G1RobotSpec:
    """Get the G1 robot specification."""
    return G1RobotSpec()


def get_reward_components_for_skill(skill_type: str) -> Dict:
    """Get relevant reward components for a skill type."""
    relevant = {}
    for name, component in REWARD_COMPONENTS.items():
        if skill_type in component["relevant_skills"] or "all" in component["relevant_skills"]:
            relevant[name] = component
    return relevant


def get_skill_template(skill_name: str) -> Optional[Dict]:
    """Get a pre-defined skill template."""
    return SKILL_TEMPLATES.get(skill_name)


def list_available_skills() -> List[str]:
    """List all available skill templates."""
    return list(SKILL_TEMPLATES.keys())
