"""
Experiment Runner

Runs structured RL training experiments for the G1 humanoid robot.
Connects the skill definitions with actual MuJoCo training.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO, SAC, TD3


def get_device(algorithm: str = "PPO", policy_type: str = "MlpPolicy"):
    """
    Get the best available device for training.

    Note: For PPO with MlpPolicy, CPU is actually faster due to the
    small network size and PPO's sequential nature. GPU is beneficial
    mainly for CNN policies or very large networks.
    """
    # For MLP policies, CPU is often faster
    if policy_type == "MlpPolicy" and algorithm in ["PPO", "A2C"]:
        return "cpu"

    # For other cases, try GPU
    if torch.cuda.is_available():
        return "cuda"
    # MPS has issues with float64, skip for now
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from src.robot.robot_spec import get_robot_spec, REWARD_COMPONENTS, SKILL_TEMPLATES


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""
    skill_id: str
    name: str
    description: str

    # Training params
    algorithm: str = "PPO"
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_envs: int = 4  # Parallel environments
    gamma: float = 0.99

    # Network architecture
    policy_type: str = "MlpPolicy"
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])

    # Reward configuration
    reward_components: List[str] = field(default_factory=list)
    reward_weights: Dict[str, float] = field(default_factory=dict)

    # Environment
    max_episode_steps: int = 1000
    target_velocity: float = 0.5  # For walking skills

    # Checkpointing
    save_freq: int = 50_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5

    # Transfer learning
    transfer_from: Optional[str] = None

    # Curriculum
    curriculum: Optional[List[Dict]] = None


class G1SkillEnv(gym.Env):
    """
    Gymnasium environment for training G1 skills.

    Features:
    - Configurable reward functions
    - Proper termination conditions
    - Observation normalization ready
    """

    def __init__(
        self,
        skill_config: ExperimentConfig,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
    ):
        super().__init__()

        self.config = skill_config
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        # Load MuJoCo model
        xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Robot spec for reference
        self.robot_spec = get_robot_spec()

        # State tracking
        self._step_count = 0
        self._prev_action = None
        self._initial_height = None

        # Get dimensions
        self.nu = self.model.nu  # Number of actuators (43)
        self.nq = self.model.nq  # Position DOF
        self.nv = self.model.nv  # Velocity DOF

        # Observation: qpos (excluding root xyz) + qvel + torso orientation
        obs_dim = (self.nq - 3) + self.nv + 9  # pos + vel + orientation matrix
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

        # Action: normalized torques for all actuators
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float64
        )

        # Renderer (note: on macOS, rendering is only available during evaluation)
        self.viewer = None
        self._render_enabled = render_mode == "human"

        # For rendering, we'll use rgb_array mode and show frames
        if self._render_enabled:
            try:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception:
                self.renderer = None
                self._render_enabled = False
        else:
            self.renderer = None

    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        # Position (exclude root xyz to make it translation invariant)
        qpos = self.data.qpos[3:].copy()  # Skip root xyz

        # Velocities
        qvel = self.data.qvel.copy()

        # Torso orientation (3x3 rotation matrix flattened)
        torso_xmat = self.data.xmat[1].reshape(9)  # Body 1 is torso

        return np.concatenate([qpos, qvel, torso_xmat])

    def _get_reward(self) -> float:
        """Compute reward based on skill configuration."""
        reward = 0.0

        # Get common quantities
        com_height = self.data.subtree_com[0][2]  # Center of mass height
        com_vel = self.data.subtree_com[0]  # COM position
        forward_vel = self.data.qvel[0]  # Root x velocity
        lateral_vel = self.data.qvel[1]  # Root y velocity

        # Torso orientation (upright = [0,0,1] in world frame)
        torso_xmat = self.data.xmat[1].reshape(3, 3)
        upright = torso_xmat[2, 2]  # z component of torso z-axis

        # Control effort
        ctrl_cost = np.sum(np.square(self.data.ctrl))

        # Action smoothness
        if self._prev_action is not None:
            smoothness_cost = np.sum(np.square(self.data.ctrl - self._prev_action))
        else:
            smoothness_cost = 0.0

        # Compute each reward component
        weights = self.config.reward_weights

        for comp_name in self.config.reward_components:
            weight = weights.get(comp_name, 1.0)

            if comp_name == "upright_reward":
                # Reward for staying upright (max 1 when perfectly upright)
                reward += weight * (upright + 1) / 2

            elif comp_name == "height_reward":
                # Reward for maintaining target height
                target_height = 1.0  # meters
                height_error = abs(com_height - target_height)
                reward += weight * np.exp(-5 * height_error)

            elif comp_name == "forward_velocity":
                # Reward for moving forward at target speed
                target_vel = self.config.target_velocity
                vel_error = abs(forward_vel - target_vel)
                reward += weight * np.exp(-2 * vel_error)

            elif comp_name == "lateral_stability":
                # Penalize lateral movement
                reward -= weight * abs(lateral_vel)

            elif comp_name == "energy_efficiency":
                # Penalize control effort
                reward -= weight * 0.001 * ctrl_cost

            elif comp_name == "smoothness":
                # Penalize jerky movements
                reward -= weight * 0.01 * smoothness_cost

            elif comp_name == "com_stability":
                # Penalize COM velocity when standing
                com_vel_norm = np.linalg.norm(self.data.qvel[:3])
                reward -= weight * 0.1 * com_vel_norm

            elif comp_name == "gait_symmetry":
                # Reward symmetric leg movements (simplified)
                left_hip = self.data.qpos[1]  # left_hip_pitch
                right_hip = self.data.qpos[7]  # right_hip_pitch
                # Ideal: 180 degrees out of phase for walking
                symmetry = np.cos(left_hip - right_hip + np.pi)
                reward += weight * (symmetry + 1) / 2

            elif comp_name == "fall_penalty":
                # Large penalty for falling (height too low)
                if com_height < 0.5:
                    reward -= weight * 10.0

            elif comp_name == "jump_height":
                # Reward for jumping (height above initial)
                if self._initial_height is not None:
                    height_gain = max(0, com_height - self._initial_height)
                    reward += weight * height_gain * 10

            elif comp_name == "landing_stability":
                # Reward for stable landing (low velocity when grounded)
                if com_height < self._initial_height + 0.05:  # Near ground
                    stability = np.exp(-np.linalg.norm(self.data.qvel[:6]))
                    reward += weight * stability

        return reward

    def _is_terminated(self) -> bool:
        """Check termination conditions."""
        com_height = self.data.subtree_com[0][2]

        # Fell down
        if com_height < 0.4:
            return True

        # Torso tilted too much
        torso_xmat = self.data.xmat[1].reshape(3, 3)
        upright = torso_xmat[2, 2]
        if upright < 0.3:  # More than ~70 degrees from vertical
            return True

        return False

    def _is_truncated(self) -> bool:
        """Check truncation (time limit)."""
        return self._step_count >= self.config.max_episode_steps

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Slightly randomize initial pose
        if seed is not None:
            np.random.seed(seed)

        # Add small random perturbations to joint positions
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, self.nq - 7)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Store initial height for jump rewards
        self._initial_height = self.data.subtree_com[0][2]
        self._step_count = 0
        self._prev_action = None

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Take a step in the environment."""
        # Scale action to actuator range
        ctrl = action * self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Update renderer if enabled
        if self._render_enabled and self.renderer is not None:
            self.renderer.update_scene(self.data)

        self._step_count += 1
        self._prev_action = action.copy()

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "com_height": self.data.subtree_com[0][2],
            "forward_velocity": self.data.qvel[0],
            "step_count": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.renderer is not None:
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self):
        """Clean up."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class ProgressCallback(BaseCallback):
    """Callback to track and display training progress."""

    def __init__(self, config: ExperimentConfig, log_file: Path, verbose: int = 1):
        super().__init__(verbose)
        self.config = config
        self.log_file = log_file
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.name}")
        print(f"Algorithm: {self.config.algorithm}")
        print(f"Total timesteps: {self.config.total_timesteps:,}")
        print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        # Collect episode info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])

        # Log progress every 10000 steps
        if self.n_calls % 10000 == 0 and self.episode_rewards:
            elapsed = time.time() - self.start_time
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0

            progress = self.n_calls / self.config.total_timesteps * 100
            fps = self.n_calls / elapsed if elapsed > 0 else 0

            print(f"Step {self.n_calls:>8,} ({progress:>5.1f}%) | "
                  f"Reward: {mean_reward:>7.2f} | "
                  f"Ep.Len: {mean_length:>6.1f} | "
                  f"FPS: {fps:>6.0f}")

            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{self.n_calls},{mean_reward},{mean_length},{elapsed}\n")

        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed/60:.1f} minutes")
        if self.episode_rewards:
            print(f"Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"{'='*60}\n")


class ExperimentRunner:
    """
    Runs training experiments for G1 robot skills.

    Features:
    - Configurable experiments
    - Parallel environments
    - Checkpointing and evaluation
    - Progress tracking
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "experiments" / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.skills_dir = PROJECT_ROOT / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def create_config_from_template(self, skill_id: str, **overrides) -> ExperimentConfig:
        """Create experiment config from a skill template."""
        template = SKILL_TEMPLATES.get(skill_id)
        if not template:
            raise ValueError(f"Unknown skill template: {skill_id}")

        config = ExperimentConfig(
            skill_id=skill_id,
            name=template["name"],
            description=template["description"],
            reward_components=template["reward_components"],
            reward_weights=template.get("reward_weights", {}),
            algorithm=template["training_config"].get("algorithm", "PPO"),
            total_timesteps=template["training_config"].get("total_timesteps", 500_000),
            learning_rate=template["training_config"].get("learning_rate", 3e-4),
            transfer_from=template["training_config"].get("transfer_from"),
            curriculum=template.get("curriculum"),
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def _make_env(self, config: ExperimentConfig, rank: int = 0, render: bool = False) -> Callable:
        """Create environment factory."""
        def _init():
            render_mode = "human" if (render and rank == 0) else None
            env = G1SkillEnv(config, render_mode=render_mode)
            env = Monitor(env)
            return env
        return _init

    def run(self, config: ExperimentConfig, render: bool = False) -> Dict[str, Any]:
        """Run a training experiment."""
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / f"{config.skill_id}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "skill_id": config.skill_id,
                "name": config.name,
                "algorithm": config.algorithm,
                "total_timesteps": config.total_timesteps,
                "reward_components": config.reward_components,
                "reward_weights": config.reward_weights,
            }, f, indent=2)

        # Create vectorized environment
        # Use single env with rendering, or multiple parallel envs for speed
        if render:
            env = DummyVecEnv([self._make_env(config, 0, render=True)])
        elif config.n_envs > 1:
            env = SubprocVecEnv([self._make_env(config, i) for i in range(config.n_envs)])
        else:
            env = DummyVecEnv([self._make_env(config)])

        # Normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Create or load model
        model = None
        if config.transfer_from:
            transfer_path = self.skills_dir / "trained" / config.transfer_from / "model.zip"
            if transfer_path.exists():
                print(f"Loading pretrained model from: {config.transfer_from}")
                AlgClass = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[config.algorithm]
                model = AlgClass.load(str(transfer_path), env=env)

        if model is None:
            # Create new model
            AlgClass = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[config.algorithm]

            policy_kwargs = {
                "net_arch": config.hidden_sizes,
            }

            device = get_device(config.algorithm, config.policy_type)
            print(f"Using device: {device}")

            model = AlgClass(
                config.policy_type,
                env,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                verbose=0,
                device=device,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(exp_dir / "tensorboard"),
            )

        # Setup callbacks
        callbacks = [
            ProgressCallback(config, exp_dir / "progress.csv"),
            CheckpointCallback(
                save_freq=config.save_freq // config.n_envs,
                save_path=str(exp_dir / "checkpoints"),
                name_prefix="model",
            ),
        ]

        # Train
        print(f"\nStarting training for: {config.name}")
        print(f"Output directory: {exp_dir}")
        print(f"TensorBoard: tensorboard --logdir {exp_dir / 'tensorboard'}")

        try:
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            # Save final model
            model_path = self.skills_dir / "trained" / config.skill_id
            model_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path / "model"))
            env.save(str(model_path / "vec_normalize.pkl"))

            print(f"\nModel saved to: {model_path}")

            # Save metrics
            metrics = {
                "skill_id": config.skill_id,
                "total_timesteps": config.total_timesteps,
                "training_time": time.time(),
            }

            with open(model_path / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        finally:
            env.close()

    def evaluate(
        self,
        skill_id: str,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a trained skill."""
        model_path = self.skills_dir / "trained" / skill_id / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found for: {skill_id}")

        # Load model
        model = PPO.load(str(model_path))

        # Create config for environment
        config = self.create_config_from_template(skill_id)

        # Create environment
        env = G1SkillEnv(config, render_mode="human" if render else None)

        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}")

        env.close()

        results = {
            "skill_id": skill_id,
            "n_episodes": n_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
        }

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f}")

        return results


def train_skill(skill_id: str, timesteps: Optional[int] = None, render: bool = False, **kwargs):
    """Convenience function to train a skill."""
    runner = ExperimentRunner()
    config = runner.create_config_from_template(skill_id, **kwargs)
    if timesteps:
        config.total_timesteps = timesteps
    return runner.run(config, render=render)


def evaluate_skill(skill_id: str, n_episodes: int = 10, render: bool = False):
    """Convenience function to evaluate a skill."""
    runner = ExperimentRunner()
    return runner.evaluate(skill_id, n_episodes, render)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or evaluate G1 robot skills")
    parser.add_argument("action", choices=["train", "eval", "list"], help="Action to perform")
    parser.add_argument("--skill", type=str, help="Skill ID to train/evaluate")
    parser.add_argument("--timesteps", type=int, help="Training timesteps (overrides default)")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")

    args = parser.parse_args()

    if args.action == "list":
        print("\nAvailable skill templates:")
        for skill_id, template in SKILL_TEMPLATES.items():
            print(f"  {skill_id}: {template['description']}")
        print()

    elif args.action == "train":
        if not args.skill:
            print("Error: --skill required for training")
            sys.exit(1)
        train_skill(args.skill, args.timesteps)

    elif args.action == "eval":
        if not args.skill:
            print("Error: --skill required for evaluation")
            sys.exit(1)
        evaluate_skill(args.skill, args.episodes, args.render)
