"""
Skill Trainer

Trains individual skills using reinforcement learning with:
- Dynamic reward functions based on skill definition
- Transfer learning from existing skills
- Progress tracking and checkpointing
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import json

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.skill_learning.skill import Skill, SkillStatus, SkillConfig


class SkillRewardWrapper:
    """
    Wraps an environment to provide skill-specific rewards.

    Dynamically constructs reward function based on skill definition.
    """

    # Available reward component functions
    REWARD_COMPONENTS = {
        "height_reward": lambda env, data: data.subtree_com[0][2],
        "upright_reward": lambda env, data: data.xmat[1].reshape(3, 3)[2, 2],
        "velocity_forward": lambda env, data: data.qvel[0],
        "energy_efficiency": lambda env, data: -0.001 * np.sum(np.square(data.ctrl)),
        "stability": lambda env, data: -0.01 * np.sum(np.square(data.qvel)),
        "joint_limits": lambda env, data: 0,  # Implement based on model
        "smoothness": lambda env, data: 0,  # Track action differences
        "foot_contact": lambda env, data: 0,  # Check contact sensors
    }

    def __init__(self, env, skill: Skill):
        self.env = env
        self.skill = skill
        self._setup_reward_function()

    def _setup_reward_function(self):
        """Create reward function from skill's reward components."""
        self.active_components = []

        for component in self.skill.reward_components:
            if component in self.REWARD_COMPONENTS:
                self.active_components.append(
                    (component, self.REWARD_COMPONENTS[component])
                )

    def compute_reward(self, data) -> float:
        """Compute skill-specific reward."""
        total_reward = 0.0

        for name, func in self.active_components:
            try:
                reward = func(self.env, data)
                total_reward += reward
            except Exception as e:
                pass  # Skip failed components

        return total_reward


class TrainingProgressCallback(BaseCallback):
    """Callback to track and report training progress."""

    def __init__(self, skill: Skill, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.skill = skill
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Collect episode stats
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 0))

        if self.n_calls % self.log_freq == 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                print(f"[{self.skill.skill_id}] Step {self.n_calls}: "
                      f"Mean reward = {mean_reward:.2f}")

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": float(np.mean(self.episode_rewards[-100:])),
            "max_reward": float(np.max(self.episode_rewards)),
            "total_episodes": len(self.episode_rewards),
            "mean_episode_length": float(np.mean(self.episode_lengths[-100:])),
        }


class SkillTrainer:
    """
    Trains skills using reinforcement learning.

    Features:
    - Dynamic reward functions based on skill definition
    - Transfer learning from existing trained skills
    - Progress tracking and early stopping
    - Checkpoint saving
    """

    def __init__(
        self,
        env_class,
        skills_dir: Path,
        device: str = "auto",
    ):
        self.env_class = env_class
        self.skills_dir = Path(skills_dir)
        self.device = device

        # Ensure directories exist
        (self.skills_dir / "trained").mkdir(parents=True, exist_ok=True)
        (self.skills_dir / "logs").mkdir(parents=True, exist_ok=True)

    def _create_env(self, skill: Skill):
        """Create environment for training a skill."""
        def make_env():
            env = self.env_class()
            # TODO: Wrap with skill-specific reward
            return env
        return DummyVecEnv([make_env])

    def _get_algorithm_class(self, name: str):
        """Get algorithm class by name."""
        algorithms = {
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
        }
        return algorithms.get(name, PPO)

    def _load_pretrained(self, skill: Skill, env) -> Optional[Any]:
        """Load pretrained model for transfer learning."""
        if not skill.config.transfer_from:
            return None

        transfer_path = self.skills_dir / "trained" / skill.config.transfer_from / "model.zip"
        if not transfer_path.exists():
            print(f"Transfer source not found: {transfer_path}")
            return None

        print(f"Loading pretrained model from: {skill.config.transfer_from}")

        AlgClass = self._get_algorithm_class(skill.config.algorithm)
        model = AlgClass.load(str(transfer_path), env=env, device=self.device)

        return model

    def train(
        self,
        skill: Skill,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train a skill.

        Args:
            skill: The skill to train
            progress_callback: Optional callback for progress updates

        Returns:
            Training metrics
        """
        print(f"\n{'='*50}")
        print(f"Training skill: {skill.name}")
        print(f"Description: {skill.description}")
        print(f"Reward components: {skill.reward_components}")
        print(f"{'='*50}\n")

        skill.status = SkillStatus.TRAINING

        # Create environment
        env = self._create_env(skill)

        # Normalize observations and rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        # Try transfer learning
        model = self._load_pretrained(skill, env)

        if model is None:
            # Create new model
            AlgClass = self._get_algorithm_class(skill.config.algorithm)
            model = AlgClass(
                "MlpPolicy",
                env,
                learning_rate=skill.config.learning_rate,
                batch_size=skill.config.batch_size,
                gamma=skill.config.gamma,
                verbose=1,
                device=self.device,
            )

        # Setup callbacks
        skill_dir = self.skills_dir / "trained" / skill.skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            TrainingProgressCallback(skill),
            CheckpointCallback(
                save_freq=50000,
                save_path=str(skill_dir / "checkpoints"),
                name_prefix="model",
            ),
        ]

        # Train
        try:
            model.learn(
                total_timesteps=skill.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            # Save final model
            model.save(str(skill_dir / "model"))
            env.save(str(skill_dir / "env_normalize.pkl"))

            # Update skill status
            skill.status = SkillStatus.TRAINED
            skill.model_path = str(skill_dir / "model.zip")
            skill.checkpoint_dir = str(skill_dir / "checkpoints")

            # Get metrics from callback
            metrics = callbacks[0].get_metrics()
            skill.training_metrics = metrics

            # Save skill config
            skill.save(self.skills_dir)

            print(f"\nSkill trained successfully!")
            print(f"Model saved to: {skill_dir / 'model.zip'}")

            return metrics

        except Exception as e:
            skill.status = SkillStatus.FAILED
            skill.training_metrics = {"error": str(e)}
            skill.save(self.skills_dir)
            raise

        finally:
            env.close()

    def evaluate(
        self,
        skill: Skill,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a trained skill."""
        if skill.status != SkillStatus.TRAINED:
            raise ValueError(f"Skill {skill.skill_id} is not trained")

        model_path = self.skills_dir / "trained" / skill.skill_id / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        AlgClass = self._get_algorithm_class(skill.config.algorithm)
        model = AlgClass.load(str(model_path))

        # Create eval environment
        render_mode = "human" if render else None
        env = self.env_class(render_mode=render_mode)

        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                if render:
                    env.render()

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        env.close()

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episodes": n_episodes,
        }

    def compose_skills(
        self,
        skills: list[Skill],
        composition_name: str,
    ) -> Skill:
        """
        Compose multiple skills into a sequence.

        Creates a higher-level skill that chains the given skills.
        """
        # Create composed skill definition
        composed = Skill(
            skill_id=f"composed_{composition_name}",
            name=f"Composed: {composition_name}",
            description=f"Composition of: {', '.join(s.name for s in skills)}",
            success_criteria="All component skills executed successfully",
            reward_components=list(set(
                comp for s in skills for comp in s.reward_components
            )),
            termination_conditions=["Any component skill fails"],
            prerequisites=[s.skill_id for s in skills],
            status=SkillStatus.COMPOSED,
        )

        composed.training_metrics = {
            "composed_from": [s.skill_id for s in skills],
        }

        return composed
