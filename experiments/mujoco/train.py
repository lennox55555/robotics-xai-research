"""
MuJoCo RL Training Script with W&B Experiment Tracking
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.wandb_callback import WandbCallback
from src.explainability.policy_analyzer import PolicyAnalyzer


def load_config(config_path: str = None):
    """Load configuration from yaml file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    return OmegaConf.load(config_path)


def make_env(env_name: str, seed: int = 0):
    """Create and wrap environment."""
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        return env
    return _init


def get_algorithm(name: str):
    """Get algorithm class by name."""
    algorithms = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
    }
    return algorithms[name]


def train(config_path: str = None):
    """Main training function."""
    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    # Load config
    cfg = load_config(config_path)

    # Set seeds
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)

    # Initialize W&B
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg),
            tags=cfg.logging.wandb.tags,
        )

    # Create environment
    env = DummyVecEnv([make_env(cfg.env.name, cfg.experiment.seed)])
    if cfg.env.normalize_obs or cfg.env.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=cfg.env.normalize_obs,
            norm_reward=cfg.env.normalize_reward,
        )

    # Create eval environment
    eval_env = DummyVecEnv([make_env(cfg.env.name, cfg.experiment.seed + 100)])
    if cfg.env.normalize_obs or cfg.env.normalize_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=cfg.env.normalize_obs,
            norm_reward=False,
            training=False,
        )

    # Create model
    AlgorithmClass = get_algorithm(cfg.training.algorithm)
    model = AlgorithmClass(
        cfg.model.policy,
        env,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        gamma=cfg.training.gamma,
        verbose=1,
        seed=cfg.experiment.seed,
        device="auto",
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.logging.save_frequency,
        save_path=str(PROJECT_ROOT / "checkpoints" / cfg.experiment.name),
        name_prefix="model",
    )
    callbacks.append(checkpoint_callback)

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(PROJECT_ROOT / "checkpoints" / cfg.experiment.name / "best"),
        log_path=str(PROJECT_ROOT / "logs" / cfg.experiment.name),
        eval_freq=cfg.logging.eval_frequency,
        n_eval_episodes=cfg.logging.n_eval_episodes,
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # W&B callback
    if cfg.logging.wandb.enabled:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    # Train
    print(f"Starting training: {cfg.experiment.name}")
    print(f"Environment: {cfg.env.name}")
    print(f"Algorithm: {cfg.training.algorithm}")
    print(f"Total timesteps: {cfg.training.total_timesteps}")

    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = PROJECT_ROOT / "checkpoints" / cfg.experiment.name / "final_model"
    model.save(str(final_path))
    print(f"Model saved to {final_path}")

    # Run explainability analysis
    if cfg.explainability.enabled:
        print("Running explainability analysis...")
        analyzer = PolicyAnalyzer(model, cfg.explainability.method)
        analyzer.analyze_and_log()

    # Cleanup
    if cfg.logging.wandb.enabled:
        wandb.finish()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    train(args.config)
