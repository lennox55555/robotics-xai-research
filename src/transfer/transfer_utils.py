"""
Transfer Learning Utilities

Provides utilities for transferring knowledge between skills:
- Policy network weight transfer
- Feature extractor sharing
- Progressive skill building
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.policies import ActorCriticPolicy


def get_policy_network(model) -> nn.Module:
    """Extract the policy network from a trained model."""
    if isinstance(model, PPO):
        return model.policy
    elif isinstance(model, (SAC, TD3)):
        return model.policy
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_names(model: nn.Module) -> List[str]:
    """Get names of all layers in a model."""
    return [name for name, _ in model.named_modules() if name]


class TransferManager:
    """
    Manages transfer learning between skills.

    Supports:
    - Full policy transfer
    - Partial layer transfer (e.g., only feature extractor)
    - Layer freezing
    - Progressive unfreezing
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)

    def load_source_model(
        self,
        skill_id: str,
        algorithm: str = "PPO",
    ):
        """Load a trained model as transfer source."""
        model_path = self.skills_dir / "trained" / skill_id / "model.zip"

        if not model_path.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        AlgClass = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algorithm]
        return AlgClass.load(str(model_path))

    def transfer_weights(
        self,
        source_model,
        target_model,
        layers: Optional[List[str]] = None,
        freeze: bool = False,
    ) -> Dict[str, Any]:
        """
        Transfer weights from source to target model.

        Args:
            source_model: Trained source model
            target_model: Target model to receive weights
            layers: Specific layers to transfer (None = all compatible)
            freeze: Whether to freeze transferred layers

        Returns:
            Transfer statistics
        """
        source_policy = get_policy_network(source_model)
        target_policy = get_policy_network(target_model)

        transferred = []
        skipped = []

        source_state = source_policy.state_dict()
        target_state = target_policy.state_dict()

        for name, param in source_state.items():
            # Check if we should transfer this layer
            if layers is not None:
                should_transfer = any(layer in name for layer in layers)
            else:
                should_transfer = True

            if not should_transfer:
                skipped.append(name)
                continue

            # Check if target has this layer with compatible shape
            if name in target_state and target_state[name].shape == param.shape:
                target_state[name] = param.clone()
                transferred.append(name)

                # Freeze if requested
                if freeze:
                    for target_name, target_param in target_policy.named_parameters():
                        if name in target_name:
                            target_param.requires_grad = False
            else:
                skipped.append(name)

        # Load transferred weights
        target_policy.load_state_dict(target_state)

        return {
            "transferred_layers": transferred,
            "skipped_layers": skipped,
            "num_transferred": len(transferred),
            "num_skipped": len(skipped),
            "frozen": freeze,
        }

    def progressive_unfreeze(
        self,
        model,
        layers_to_unfreeze: List[str],
    ):
        """
        Progressively unfreeze layers during training.

        Useful for fine-tuning: start with frozen features,
        then gradually unfreeze deeper layers.
        """
        policy = get_policy_network(model)

        unfrozen = []
        for name, param in policy.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
                unfrozen.append(name)

        return unfrozen

    def compute_similarity(
        self,
        model1,
        model2,
    ) -> Dict[str, float]:
        """
        Compute similarity between two models' policies.

        Useful for understanding how much a skill has diverged
        from its transfer source.
        """
        policy1 = get_policy_network(model1)
        policy2 = get_policy_network(model2)

        similarities = {}

        state1 = policy1.state_dict()
        state2 = policy2.state_dict()

        for name in state1:
            if name in state2 and state1[name].shape == state2[name].shape:
                # Cosine similarity
                flat1 = state1[name].flatten().float()
                flat2 = state2[name].flatten().float()

                cos_sim = torch.nn.functional.cosine_similarity(
                    flat1.unsqueeze(0),
                    flat2.unsqueeze(0)
                ).item()

                similarities[name] = cos_sim

        # Overall similarity
        if similarities:
            similarities["overall"] = np.mean(list(similarities.values()))

        return similarities


class SkillEmbedding:
    """
    Creates embeddings of skills for similarity comparison.

    Uses the policy network activations as skill representations.
    """

    def __init__(self, model, sample_obs: np.ndarray):
        self.model = model
        self.sample_obs = sample_obs
        self._embedding = None

    def compute_embedding(self, n_samples: int = 100) -> np.ndarray:
        """
        Compute embedding by averaging policy activations.
        """
        policy = get_policy_network(self.model)
        policy.eval()

        embeddings = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Add noise to observations
                obs = self.sample_obs + np.random.randn(*self.sample_obs.shape) * 0.1
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                # Get features from policy
                features = policy.extract_features(obs_tensor)
                embeddings.append(features.numpy().flatten())

        self._embedding = np.mean(embeddings, axis=0)
        return self._embedding

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            raise ValueError("Call compute_embedding() first")
        return self._embedding


def find_best_transfer_source(
    target_skill_description: str,
    trained_skills: List[Dict[str, Any]],
    skill_embeddings: Dict[str, np.ndarray],
) -> Optional[str]:
    """
    Find the best skill to transfer from based on:
    - Skill description similarity
    - Policy embedding similarity
    - Reward component overlap

    Returns skill_id of best transfer source.
    """
    if not trained_skills or not skill_embeddings:
        return None

    # Simple heuristic: find skill with most similar reward components
    # In practice, you'd use more sophisticated matching

    best_skill = None
    best_score = 0

    for skill in trained_skills:
        # Check if we have embedding
        if skill["skill_id"] not in skill_embeddings:
            continue

        # Score based on various factors
        score = 0

        # Reward component overlap (if available)
        # This is simplified - real implementation would be more sophisticated
        score += 1  # Placeholder

        if score > best_score:
            best_score = score
            best_skill = skill["skill_id"]

    return best_skill
