"""
Unitree G1 Humanoid Environment with Dexterous Hands

A custom Gymnasium environment wrapping the Unitree G1 humanoid robot
with articulated fingers from MuJoCo Menagerie.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class G1HumanoidEnv(gym.Env):
    """
    Unitree G1 Humanoid with dexterous hands environment.

    Observation space: Joint positions + velocities + body orientation
    Action space: Joint torques for all 43 actuators

    Task: Stand upright and maintain balance (standup task)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        frame_skip=5,
        task="standup",  # standup, walk, reach
    ):
        super().__init__()

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.task = task

        # Load MuJoCo model
        model_path = Path(__file__).parent.parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Renderer for visualization
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Action space: torques for all actuators
        self.n_actuators = self.model.nu
        action_low = -1.0 * np.ones(self.n_actuators, dtype=np.float32)
        action_high = 1.0 * np.ones(self.n_actuators, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Observation space: qpos + qvel + sensor data
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Store initial state for reset
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()

        # Viewer for human rendering
        self._viewer = None

    def _get_obs(self):
        """Get current observation."""
        # Joint positions (excluding free joint root)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Center of mass position and velocity
        com_pos = self.data.subtree_com[0].copy()

        # Body orientation (torso)
        torso_xmat = self.data.xmat[1].reshape(3, 3)

        return np.concatenate([
            qpos,
            qvel,
            com_pos,
            torso_xmat.flatten(),
        ])

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Add small random perturbation
        if seed is not None:
            np.random.seed(seed)

        noise_scale = 0.01
        self.data.qpos[:] = self._initial_qpos + noise_scale * np.random.randn(len(self._initial_qpos))
        self.data.qvel[:] = self._initial_qvel + noise_scale * np.random.randn(len(self._initial_qvel))

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute action and return new state."""
        # Scale actions to actuator control range
        action = np.clip(action, -1, 1)
        ctrl_range = self.model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + (action + 1) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

        # Apply action
        self.data.ctrl[:] = scaled_action

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """Compute reward based on task."""
        # Height of center of mass
        com_height = self.data.subtree_com[0][2]

        # Upright reward (higher is better)
        height_reward = com_height

        # Stability reward (low velocity is good)
        vel_penalty = 0.01 * np.sum(np.square(self.data.qvel))

        # Control cost
        ctrl_cost = 0.001 * np.sum(np.square(self.data.ctrl))

        reward = height_reward - vel_penalty - ctrl_cost

        return reward

    def _is_terminated(self):
        """Check if episode should terminate."""
        # Terminate if robot falls (COM too low)
        com_height = self.data.subtree_com[0][2]
        if com_height < 0.3:  # Fallen
            return True
        return False

    def _get_info(self):
        """Return additional info."""
        return {
            "com_height": self.data.subtree_com[0][2],
            "com_position": self.data.subtree_com[0].copy(),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Register the environment with Gymnasium
gym.register(
    id="G1Humanoid-v0",
    entry_point="src.envs.g1_humanoid:G1HumanoidEnv",
    max_episode_steps=1000,
)
