#!/usr/bin/env python3
"""
ATLAS Phase 5: XAI Comparison -- Specialist vs VLA Saliency Analysis

Compares what the RL specialist policies attend to vs what the VLA attends to
by computing input saliency maps over the proprioceptive observation space.

Methods:
1. Input saliency (gradient-based) for both specialists and VLA
2. Feature importance aggregated over 100 observations per skill
3. Cosine similarity between specialist and VLA saliency vectors
4. Shannon entropy of saliency distributions
5. Joint-group attribution (which body parts each model relies on)

Outputs:
- Saliency arrays (.npy) for both model types
- Statistical test results (cosine similarity, entropy, Wilcoxon)
- Publication-ready figures
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.spatial.distance import cosine as cosine_distance

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.experiments.experiment_runner import G1SkillEnv, ExperimentConfig
from src.models.simple_vla import load_vla, SKILL_TO_ID
import mujoco


# Joint group mapping for the 43-DOF G1 model
# Obs is: qpos[3:] (nq-3) + qvel (nv) + torso_xmat (9) + perturbation (6)
# We label the proprioceptive features by body group
JOINT_GROUPS = {
    "left_leg": list(range(0, 6)) + list(range(46, 52)),      # qpos + qvel indices
    "right_leg": list(range(6, 12)) + list(range(52, 58)),
    "torso": list(range(12, 15)) + list(range(58, 61)),
    "left_arm": list(range(15, 22)) + list(range(61, 68)),
    "right_arm": list(range(22, 29)) + list(range(68, 75)),
    "left_hand": list(range(29, 36)) + list(range(75, 82)),
    "right_hand": list(range(36, 43)) + list(range(82, 89)),
    "orientation": list(range(89, 98)),                         # torso_xmat 9 dims
    "root_vel": list(range(43, 46)),                            # root xyz velocity
    "perturbation": list(range(105, 111)),                      # force sensing
}

SKILLS = ["balance_stand", "squat", "jump", "walk_forward", "raise_right_hand"]
N_OBS = 100  # Observations per skill for analysis

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "atlas" / "phase5_xai"
SKILLS_DIR = PROJECT_ROOT / "skills" / "trained"


def collect_observations(skill_id: str, n_obs: int = 100) -> np.ndarray:
    """Collect observations by running the specialist policy."""
    model_path = SKILLS_DIR / skill_id / "model.zip"
    normalize_path = SKILLS_DIR / skill_id / "vec_normalize.pkl"

    policy = PPO.load(str(model_path))

    # Create env
    config = ExperimentConfig(
        skill_id=skill_id, name=skill_id, description="",
        reward_components=["upright_reward"],
        reward_weights={"upright_reward": 1.0},
    )
    env = G1SkillEnv(config)

    # Load VecNormalize for proper normalization
    vec_env = None
    if normalize_path.exists():
        dummy_env = DummyVecEnv([lambda: G1SkillEnv(config)])
        vec_env = VecNormalize.load(str(normalize_path), dummy_env)
        vec_env.training = False

    observations = []
    obs, _ = env.reset()
    for _ in range(n_obs * 10):  # Run extra steps, sample every 10th
        if vec_env:
            obs_normalized = vec_env.normalize_obs(obs)
        else:
            obs_normalized = obs

        action, _ = policy.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if len(observations) < n_obs:
            observations.append(obs_normalized.copy())

        if terminated or truncated:
            obs, _ = env.reset()

        if len(observations) >= n_obs:
            break

    env.close()
    return np.array(observations)


def compute_specialist_saliency(skill_id: str, observations: np.ndarray) -> np.ndarray:
    """Compute saliency for the specialist PPO policy."""
    model_path = SKILLS_DIR / skill_id / "model.zip"
    model = PPO.load(str(model_path))
    policy = model.policy
    device = next(policy.parameters()).device

    saliencies = []
    for obs in observations:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        obs_tensor.requires_grad_(True)

        with torch.enable_grad():
            action, value, _ = policy(obs_tensor)
            value.backward()
            sal = obs_tensor.grad.abs().cpu().numpy().squeeze()
            saliencies.append(sal)

    return np.array(saliencies)


def compute_vla_saliency(vla_model, skill_id: str, observations: np.ndarray) -> np.ndarray:
    """Compute saliency of VLA's action output w.r.t. proprioceptive input."""
    skill_idx = SKILL_TO_ID.get(skill_id, 0)
    device = next(vla_model.parameters()).device

    # Create a dummy image (we're measuring proprio saliency, not vision)
    dummy_image = torch.zeros(1, 3, 128, 128).to(device)
    skill_tensor = torch.LongTensor([skill_idx]).to(device)

    saliencies = []
    for obs in observations:
        proprio = torch.FloatTensor(obs).unsqueeze(0).to(device)
        proprio.requires_grad_(True)

        with torch.enable_grad():
            action = vla_model(dummy_image, proprio, skill_tensor)
            # Use L2 norm of action as scalar to backprop
            action_norm = action.norm()
            action_norm.backward()
            sal = proprio.grad.abs().cpu().numpy().squeeze()
            saliencies.append(sal)

    return np.array(saliencies)


def compute_cosine_similarity(sal_a: np.ndarray, sal_b: np.ndarray) -> np.ndarray:
    """Compute per-observation cosine similarity between two saliency arrays."""
    similarities = []
    for a, b in zip(sal_a, sal_b):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            similarities.append(0.0)
        else:
            similarities.append(1.0 - cosine_distance(a, b))
    return np.array(similarities)


def compute_entropy(saliency: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy of normalized saliency distributions."""
    entropies = []
    for sal in saliency:
        # Normalize to probability distribution
        total = sal.sum()
        if total == 0:
            entropies.append(0.0)
            continue
        p = sal / total
        p = p[p > 0]  # Remove zeros for log
        entropies.append(-np.sum(p * np.log2(p)))
    return np.array(entropies)


def compute_joint_group_attribution(saliency: np.ndarray) -> Dict[str, float]:
    """Aggregate saliency by joint group."""
    mean_sal = saliency.mean(axis=0)
    total = mean_sal.sum()
    if total == 0:
        return {group: 0.0 for group in JOINT_GROUPS}

    attribution = {}
    for group, indices in JOINT_GROUPS.items():
        valid_indices = [i for i in indices if i < len(mean_sal)]
        if valid_indices:
            attribution[group] = float(mean_sal[valid_indices].sum() / total)
        else:
            attribution[group] = 0.0
    return attribution


def plot_feature_importance_comparison(
    specialist_sal: np.ndarray,
    vla_sal: np.ndarray,
    skill_id: str,
    output_path: Path,
):
    """Side-by-side feature importance bar chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    spec_mean = specialist_sal.mean(axis=0)
    vla_mean = vla_sal.mean(axis=0)
    x = np.arange(len(spec_mean))

    ax1.bar(x, spec_mean, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Saliency')
    ax1.set_title(f'{skill_id} -- RL Specialist')

    ax2.bar(x, vla_mean, color='coral', alpha=0.8)
    ax2.set_ylabel('Saliency')
    ax2.set_xlabel('Feature Index')
    ax2.set_title(f'{skill_id} -- VLA')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_joint_group_comparison(
    all_specialist_attr: Dict[str, Dict[str, float]],
    all_vla_attr: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Joint-group attribution bar chart across all skills."""
    groups = list(JOINT_GROUPS.keys())
    n_skills = len(SKILLS)
    n_groups = len(groups)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Specialist
    data_spec = np.zeros((n_skills, n_groups))
    for i, skill in enumerate(SKILLS):
        for j, group in enumerate(groups):
            data_spec[i, j] = all_specialist_attr[skill].get(group, 0)

    im1 = axes[0].imshow(data_spec, aspect='auto', cmap='Blues')
    axes[0].set_xticks(range(n_groups))
    axes[0].set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(n_skills))
    axes[0].set_yticklabels(SKILLS, fontsize=9)
    axes[0].set_title('RL Specialists -- Joint Group Attribution')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # VLA
    data_vla = np.zeros((n_skills, n_groups))
    for i, skill in enumerate(SKILLS):
        for j, group in enumerate(groups):
            data_vla[i, j] = all_vla_attr[skill].get(group, 0)

    im2 = axes[1].imshow(data_vla, aspect='auto', cmap='Oranges')
    axes[1].set_xticks(range(n_groups))
    axes[1].set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(n_skills))
    axes[1].set_yticklabels(SKILLS, fontsize=9)
    axes[1].set_title('VLA -- Joint Group Attribution')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_saliency_divergence(all_cosine_sims: Dict[str, np.ndarray], output_path: Path):
    """Boxplot of cosine similarities across skills."""
    fig, ax = plt.subplots(figsize=(10, 5))

    data = [all_cosine_sims[skill] for skill in SKILLS]
    bp = ax.boxplot(data, labels=SKILLS, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='H2 threshold (0.7)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Specialist vs VLA Saliency Divergence')
    ax.legend()

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_entropy_comparison(
    spec_entropies: Dict[str, np.ndarray],
    vla_entropies: Dict[str, np.ndarray],
    output_path: Path,
):
    """Bar chart comparing specialist vs VLA entropy per skill."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(SKILLS))
    width = 0.35

    spec_means = [spec_entropies[s].mean() for s in SKILLS]
    spec_stds = [spec_entropies[s].std() for s in SKILLS]
    vla_means = [vla_entropies[s].mean() for s in SKILLS]
    vla_stds = [vla_entropies[s].std() for s in SKILLS]

    ax.bar(x - width/2, spec_means, width, yerr=spec_stds, label='Specialist', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, vla_means, width, yerr=vla_stds, label='VLA', color='coral', alpha=0.8)

    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Saliency Distribution Entropy: Specialist vs VLA')
    ax.set_xticks(x)
    ax.set_xticklabels(SKILLS, rotation=15)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("ATLAS Phase 5: XAI Comparison")
    print("=" * 60)

    # Create output dirs
    for subdir in ["saliency/specialist", "saliency/vla", "feature_importance/specialist",
                   "feature_importance/vla", "figures"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Load VLA
    print("\nLoading VLA model...")
    vla_model = load_vla()
    vla_model.eval()
    print(f"VLA: {sum(p.numel() for p in vla_model.parameters()):,} parameters")

    all_cosine_sims = {}
    all_spec_entropies = {}
    all_vla_entropies = {}
    all_spec_attr = {}
    all_vla_attr = {}

    for skill_id in SKILLS:
        print(f"\n{'='*40}")
        print(f"SKILL: {skill_id}")
        print(f"{'='*40}")

        # Step 1: Collect observations
        print(f"  Collecting {N_OBS} observations...")
        observations = collect_observations(skill_id, N_OBS)
        print(f"  Observations shape: {observations.shape}")

        # Step 2: Specialist saliency
        print(f"  Computing specialist saliency...")
        spec_sal = compute_specialist_saliency(skill_id, observations)
        np.save(OUTPUT_DIR / "saliency" / "specialist" / f"{skill_id}.npy", spec_sal)

        # Step 3: VLA saliency
        print(f"  Computing VLA saliency...")
        vla_sal = compute_vla_saliency(vla_model, skill_id, observations)
        np.save(OUTPUT_DIR / "saliency" / "vla" / f"{skill_id}.npy", vla_sal)

        # Step 4: Cosine similarity
        cos_sims = compute_cosine_similarity(spec_sal, vla_sal)
        all_cosine_sims[skill_id] = cos_sims
        print(f"  Cosine similarity: {cos_sims.mean():.3f} +/- {cos_sims.std():.3f}")

        # Step 5: Entropy
        spec_ent = compute_entropy(spec_sal)
        vla_ent = compute_entropy(vla_sal)
        all_spec_entropies[skill_id] = spec_ent
        all_vla_entropies[skill_id] = vla_ent
        print(f"  Specialist entropy: {spec_ent.mean():.3f} +/- {spec_ent.std():.3f}")
        print(f"  VLA entropy: {vla_ent.mean():.3f} +/- {vla_ent.std():.3f}")

        # Step 6: Joint group attribution
        spec_attr = compute_joint_group_attribution(spec_sal)
        vla_attr = compute_joint_group_attribution(vla_sal)
        all_spec_attr[skill_id] = spec_attr
        all_vla_attr[skill_id] = vla_attr

        # Step 7: Per-skill figure
        plot_feature_importance_comparison(
            spec_sal, vla_sal, skill_id,
            OUTPUT_DIR / "figures" / f"feature_importance_{skill_id}.png",
        )
        print(f"  Figure saved")

    # Statistical tests
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS")
    print(f"{'='*60}")

    test_results = {}
    for skill_id in SKILLS:
        cos_sims = all_cosine_sims[skill_id]
        spec_ent = all_spec_entropies[skill_id]
        vla_ent = all_vla_entropies[skill_id]

        # Wilcoxon signed-rank test on entropy
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(spec_ent, vla_ent)

        # Bootstrap 95% CI on cosine similarity mean
        boot_means = []
        for _ in range(1000):
            sample = np.random.choice(cos_sims, size=len(cos_sims), replace=True)
            boot_means.append(sample.mean())
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        test_results[skill_id] = {
            "cosine_sim_mean": float(cos_sims.mean()),
            "cosine_sim_std": float(cos_sims.std()),
            "cosine_sim_ci95": [float(ci_low), float(ci_high)],
            "spec_entropy_mean": float(spec_ent.mean()),
            "vla_entropy_mean": float(vla_ent.mean()),
            "wilcoxon_statistic": float(wilcoxon_stat),
            "wilcoxon_p_value": float(wilcoxon_p),
            "wilcoxon_p_bonferroni": float(min(wilcoxon_p * len(SKILLS), 1.0)),
        }

        print(f"\n{skill_id}:")
        print(f"  Cosine sim: {cos_sims.mean():.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  Entropy -- Specialist: {spec_ent.mean():.3f}, VLA: {vla_ent.mean():.3f}")
        print(f"  Wilcoxon p={wilcoxon_p:.4f} (Bonferroni: {min(wilcoxon_p * len(SKILLS), 1.0):.4f})")

    # Save results
    results_path = OUTPUT_DIR / "statistical_tests.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cross-skill figures
    print("\nGenerating cross-skill figures...")
    plot_saliency_divergence(all_cosine_sims, OUTPUT_DIR / "figures" / "saliency_divergence.png")
    plot_entropy_comparison(all_spec_entropies, all_vla_entropies, OUTPUT_DIR / "figures" / "entropy_comparison.png")
    plot_joint_group_comparison(all_spec_attr, all_vla_attr, OUTPUT_DIR / "figures" / "joint_group_attribution.png")

    # Save attribution data
    attr_data = {
        "specialist": {s: all_spec_attr[s] for s in SKILLS},
        "vla": {s: all_vla_attr[s] for s in SKILLS},
    }
    with open(OUTPUT_DIR / "feature_importance" / "joint_group_attribution.json", "w") as f:
        json.dump(attr_data, f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 5 COMPLETE")
    print(f"{'='*60}")
    print(f"Figures: {OUTPUT_DIR / 'figures'}")
    print(f"Stats: {results_path}")


if __name__ == "__main__":
    main()
