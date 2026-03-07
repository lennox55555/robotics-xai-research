"""
Policy Explainability/Interpretability Analysis Tools
"""

import numpy as np
import torch
import wandb
from typing import Optional, List, Dict, Any


class PolicyAnalyzer:
    """
    Analyzes trained RL policies for explainability and interpretability.

    Supports multiple analysis methods:
    - attention: Attention weight visualization (for attention-based policies)
    - gradcam: Gradient-weighted class activation mapping
    - shap: SHAP value analysis for feature importance
    - saliency: Input gradient saliency maps
    """

    def __init__(self, model, method: str = "saliency"):
        self.model = model
        self.method = method
        self.policy = model.policy
        self.device = next(self.policy.parameters()).device

    def compute_saliency(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute saliency map showing input feature importance.

        Args:
            obs: Observation array

        Returns:
            Saliency values for each input dimension
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        obs_tensor.requires_grad_(True)

        # Forward pass
        with torch.enable_grad():
            action, value, _ = self.policy(obs_tensor)

            # Compute gradient of value w.r.t. input
            value.backward()

            saliency = obs_tensor.grad.abs().cpu().numpy().squeeze()

        return saliency

    def compute_feature_importance(
        self,
        observations: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute aggregate feature importance over multiple observations.

        Args:
            observations: Array of observations
            n_samples: Number of samples to analyze

        Returns:
            Dictionary with importance metrics
        """
        indices = np.random.choice(len(observations), min(n_samples, len(observations)), replace=False)

        saliencies = []
        for idx in indices:
            sal = self.compute_saliency(observations[idx])
            saliencies.append(sal)

        saliencies = np.array(saliencies)

        return {
            "mean_importance": saliencies.mean(axis=0),
            "std_importance": saliencies.std(axis=0),
            "max_importance": saliencies.max(axis=0),
        }

    def analyze_action_distribution(
        self,
        obs: np.ndarray,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Analyze the action distribution for a given observation.

        Args:
            obs: Observation to analyze
            n_samples: Number of action samples

        Returns:
            Dictionary with distribution statistics
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distribution = self.policy.get_distribution(obs_tensor)
            actions = distribution.sample((n_samples,))

            if hasattr(distribution, "entropy"):
                entropy = distribution.entropy().item()
            else:
                entropy = None

        actions_np = actions.cpu().numpy().squeeze()

        return {
            "action_mean": actions_np.mean(axis=0),
            "action_std": actions_np.std(axis=0),
            "action_min": actions_np.min(axis=0),
            "action_max": actions_np.max(axis=0),
            "entropy": entropy,
        }

    def analyze_and_log(
        self,
        observations: Optional[np.ndarray] = None,
        log_to_wandb: bool = True
    ) -> Dict[str, Any]:
        """
        Run full explainability analysis and optionally log to W&B.

        Args:
            observations: Optional observations to analyze
            log_to_wandb: Whether to log results to W&B

        Returns:
            Analysis results dictionary
        """
        results = {}

        if observations is not None:
            # Feature importance analysis
            importance = self.compute_feature_importance(observations)
            results["feature_importance"] = importance

            if log_to_wandb and wandb.run is not None:
                # Create feature importance bar chart
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 6))
                n_features = len(importance["mean_importance"])
                x = np.arange(n_features)

                ax.bar(x, importance["mean_importance"], yerr=importance["std_importance"])
                ax.set_xlabel("Feature Index")
                ax.set_ylabel("Importance (Mean Saliency)")
                ax.set_title("Feature Importance Analysis")

                wandb.log({"explainability/feature_importance": wandb.Image(fig)})
                plt.close(fig)

        # Log model architecture info
        if log_to_wandb and wandb.run is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.policy.parameters())
            trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

            wandb.log({
                "explainability/total_params": total_params,
                "explainability/trainable_params": trainable_params,
            })

        return results
