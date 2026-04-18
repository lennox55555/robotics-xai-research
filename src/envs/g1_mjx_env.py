"""
G1 Humanoid MJX Environment

JAX-native vectorized environment using MuJoCo MJX for GPU/TPU-accelerated
training. Runs thousands of parallel simulations through JIT-compiled code.

Uses the simplified G1 model (29 actuators, no finger joints) from
mujoco_menagerie/unitree_g1/scene_mjx.xml.

Architecture:
  - MJX env: fast training (1024+ parallel envs, ~12k steps/sec on M4 Max)
  - G1SkillEnv: rendering, video recording, trajectory collection, web UI
"""

import os
# Force CPU backend -- Metal GPU is not yet supported by MJX
# (JAX-Metal throws UNIMPLEMENTED: default_memory_space)
os.environ["JAX_PLATFORMS"] = "cpu"

import functools
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_mjx_model(xml_path: Optional[str] = None):
    """Load the G1 MJX model and put it on device."""
    if xml_path is None:
        xml_path = str(PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_mjx.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


def make_initial_state(mj_model, mjx_model, rng: jax.Array):
    """Create a single initial MJX state with slight randomization."""
    mj_data = mujoco.MjData(mj_model)

    # Use the "home" keyframe if available
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    mjx_data = mjx.put_data(mj_model, mj_data)

    # Add small random perturbation to joint positions (skip root 7 qpos)
    noise = jax.random.uniform(rng, shape=(mj_model.nq - 7,), minval=-0.02, maxval=0.02)
    qpos = mjx_data.qpos.at[7:].add(noise)
    mjx_data = mjx_data.replace(qpos=qpos)

    return mjx_data


def get_obs(mjx_data, nq: int) -> jax.Array:
    """Extract observation vector from MJX data.

    Matches G1SkillEnv: qpos[3:] + qvel + torso_xmat(flattened 9).
    For MJX model: torso_link is body index 16.
    """
    qpos = mjx_data.qpos[3:]  # skip root xyz
    qvel = mjx_data.qvel
    # torso_link is body 16 in the MJX model
    torso_xmat = mjx_data.xmat[16].reshape(9)
    return jnp.concatenate([qpos, qvel, torso_xmat])


def compute_reward(
    mjx_data,
    prev_action: jax.Array,
    initial_height: float,
    reward_components: List[str],
    reward_weights: Dict[str, float],
    target_velocity: float = 0.5,
) -> jax.Array:
    """Compute reward from MJX data. JAX-compatible (no Python control flow on data).

    Body indices for MJX G1 model:
      0: world, 1: pelvis, 7: left_ankle_roll_link, 13: right_ankle_roll_link,
      16: torso_link, 23: left_wrist_yaw_link, 30: right_wrist_yaw_link
    """
    reward = jnp.float32(0.0)

    com_height = mjx_data.subtree_com[0][2]
    forward_vel = mjx_data.qvel[0]
    lateral_vel = mjx_data.qvel[1]
    torso_xmat = mjx_data.xmat[16].reshape(3, 3)
    upright = torso_xmat[2, 2]
    ctrl_cost = jnp.sum(jnp.square(mjx_data.ctrl))

    for comp_name in reward_components:
        w = jnp.float32(reward_weights.get(comp_name, 1.0))

        if comp_name == "upright_reward":
            reward += w * (upright + 1) / 2

        elif comp_name == "height_reward":
            height_error = jnp.abs(com_height - 1.0)
            reward += w * jnp.exp(-5 * height_error)

        elif comp_name == "forward_velocity":
            vel_error = jnp.abs(forward_vel - target_velocity)
            reward += w * jnp.exp(-2 * vel_error)

        elif comp_name == "lateral_stability":
            reward -= w * jnp.abs(lateral_vel)

        elif comp_name == "energy_efficiency":
            reward -= w * 0.001 * ctrl_cost

        elif comp_name == "smoothness":
            smoothness_cost = jnp.sum(jnp.square(mjx_data.ctrl - prev_action))
            reward -= w * 0.01 * smoothness_cost

        elif comp_name == "com_stability":
            com_vel_norm = jnp.linalg.norm(mjx_data.qvel[:3])
            reward -= w * 0.1 * com_vel_norm

        elif comp_name == "gait_symmetry":
            left_hip = mjx_data.qpos[7]   # after root 7 qpos
            right_hip = mjx_data.qpos[13]
            symmetry = jnp.cos(left_hip - right_hip + jnp.pi)
            reward += w * (symmetry + 1) / 2

        elif comp_name == "fall_penalty":
            reward -= w * 10.0 * (com_height < 0.5).astype(jnp.float32)

        elif comp_name == "jump_height":
            height_gain = jnp.maximum(0.0, com_height - initial_height)
            reward += w * height_gain * 10

        elif comp_name == "landing_stability":
            near_ground = (com_height < initial_height + 0.05).astype(jnp.float32)
            stability = jnp.exp(-jnp.linalg.norm(mjx_data.qvel[:6]))
            reward += w * near_ground * stability

        elif comp_name == "right_hand_height":
            # right_wrist_yaw_link is body 30
            hand_height = mjx_data.xpos[30][2]
            reward += w * jnp.minimum(hand_height / 1.5, 1.5)

        elif comp_name == "left_hand_height":
            # left_wrist_yaw_link is body 23
            hand_height = mjx_data.xpos[23][2]
            reward += w * jnp.minimum(hand_height / 1.5, 1.5)

        elif comp_name == "wave_motion":
            hand_height = mjx_data.xpos[30][2]
            hand_vel_y = jnp.abs(mjx_data.cvel[30][4])
            wave_reward = jnp.where(hand_height > 1.2, jnp.minimum(hand_vel_y, 2.0), 0.0)
            reward += w * wave_reward

        elif comp_name == "yaw_rotation":
            yaw_vel = mjx_data.qvel[5]
            reward += w * yaw_vel

        elif comp_name == "squat_depth":
            # Reward for lowering COM to target squat height
            target_squat_height = 0.55  # meters (deep squat)
            squat_error = jnp.abs(com_height - target_squat_height)
            reward += w * jnp.exp(-5 * squat_error)

    return reward


def is_terminated(mjx_data, min_com_height: float = 0.4, min_upright: float = 0.3) -> jax.Array:
    """Check termination. Returns boolean JAX array."""
    com_height = mjx_data.subtree_com[0][2]
    upright = mjx_data.xmat[16].reshape(3, 3)[2, 2]
    return (com_height < min_com_height) | (upright < min_upright)


class MJXTrainer:
    """Trains RL policies using MJX-accelerated parallel simulation.

    Uses JAX's vmap to run thousands of environments in parallel,
    with JIT-compiled step and reward functions.
    """

    def __init__(
        self,
        skill_config: dict,
        num_envs: int = 1024,
        max_episode_steps: int = 1000,
        frame_skip: int = 5,
    ):
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.skill_config = skill_config

        # Load model
        self.mj_model, self.mjx_model = load_mjx_model()
        self.nu = self.mj_model.nu  # 29 actuators
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv

        # Observation dim: (nq - 3) + nv + 9
        self.obs_dim = (self.nq - 3) + self.nv + 9

        # Reward config
        self.reward_components = skill_config.get("reward_components", ["upright_reward", "height_reward"])
        self.reward_weights = skill_config.get("reward_weights", {})
        self.target_velocity = skill_config.get("target_velocity", 0.5)
        self.min_com_height = skill_config.get("min_com_height", 0.4)
        self.min_upright = skill_config.get("min_upright", 0.3)

        # Build JIT-compiled functions
        self._build_jit_functions()

    def _build_jit_functions(self):
        """Pre-compile all JAX functions."""
        mjx_model = self.mjx_model
        frame_skip = self.frame_skip
        nq = self.nq
        reward_components = self.reward_components
        reward_weights = self.reward_weights
        target_velocity = self.target_velocity
        min_com_height = self.min_com_height
        min_upright = self.min_upright

        @jax.jit
        def step_single(mjx_data, action, prev_action, initial_height):
            """Step a single environment."""
            # Scale action to control range
            ctrl = action * mjx_model.actuator_ctrlrange[:, 1]
            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Physics substeps
            def substep(data, _):
                return mjx.step(mjx_model, data), None
            mjx_data, _ = jax.lax.scan(substep, mjx_data, None, length=frame_skip)

            obs = get_obs(mjx_data, nq)
            reward = compute_reward(
                mjx_data, prev_action, initial_height,
                reward_components, reward_weights, target_velocity,
            )
            done = is_terminated(mjx_data, min_com_height, min_upright)

            return mjx_data, obs, reward, done

        @jax.jit
        def batched_step(batch_data, actions, prev_actions, initial_heights):
            """Step all environments in parallel."""
            return jax.vmap(step_single)(batch_data, actions, prev_actions, initial_heights)

        self.step_single = step_single
        self.batched_step = batched_step

    def reset_all(self, rng: jax.Array):
        """Reset all environments. Returns batched MJX data and observations."""
        rngs = jax.random.split(rng, self.num_envs)
        batch_data = jax.vmap(
            lambda r: make_initial_state(self.mj_model, self.mjx_model, r)
        )(rngs)

        batch_obs = jax.vmap(lambda d: get_obs(d, self.nq))(batch_data)
        initial_heights = batch_data.subtree_com[:, 0, 2]

        return batch_data, batch_obs, initial_heights

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.nu
