"""
W&B Callback for Stable-Baselines3
"""

import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    """
    Custom callback for logging training metrics to Weights & Biases.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if "r" in ep_info:
                wandb.log({
                    "rollout/ep_rew_mean": ep_info["r"],
                    "rollout/ep_len_mean": ep_info["l"],
                    "time/total_timesteps": self.num_timesteps,
                }, step=self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        # Log additional metrics at end of rollout
        if hasattr(self.model, "logger") and self.model.logger is not None:
            # Get latest logged values
            logs = {}

            if hasattr(self.model, "policy"):
                # Log policy entropy if available
                if hasattr(self.model.policy, "entropy"):
                    logs["train/entropy"] = float(self.model.policy.entropy)

            if logs:
                wandb.log(logs, step=self.num_timesteps)
