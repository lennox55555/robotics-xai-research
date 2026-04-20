#!/usr/bin/env python3
"""
ATLAS Phase 3: Collect trajectory demonstrations from all 6 specialists.

For each trained specialist, rolls out episodes and records:
- Proprioceptive observations (111-dim)
- Actions (43-dim)
- Rewards
- Camera images (128x128 RGB from track camera)
- Language commands (4 phrasings per skill)

Output: HDF5 files in experiments/atlas/phase3_trajectories/raw/
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.experiments.experiment_runner import G1SkillEnv, ExperimentConfig
from src.utils.trajectory_recorder import TrajectoryRecorder

SKILLS_DIR = PROJECT_ROOT / "skills" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "atlas" / "phase3_trajectories" / "raw"

# Language commands per skill (4 phrasings each for diversity)
LANGUAGE_COMMANDS = {
    "balance_stand": [
        "stand still and balance",
        "maintain upright standing position",
        "balance without moving",
        "hold a stable standing pose",
    ],
    "squat": [
        "squat down",
        "lower into a squat position",
        "crouch down and hold",
        "perform a deep squat",
    ],
    "jump": [
        "jump up",
        "perform a vertical jump",
        "jump and land",
        "leap upward",
    ],
    "walk_forward": [
        "walk forward",
        "move forward at a moderate pace",
        "take steps forward",
        "walk straight ahead",
    ],
    "raise_right_hand": [
        "raise your right hand",
        "lift the right hand above your head",
        "put your right hand up",
        "raise right arm",
    ],
    "wave_right_hand": [
        "wave your right hand",
        "wave hello with your right hand",
        "wave at me",
        "wave goodbye",
    ],
}

# Collection config per skill
COLLECTION_CONFIG = {
    "balance_stand": {"num_episodes": 2000, "min_survival": 50},
    "squat": {"num_episodes": 2000, "min_survival": 50},
    "jump": {"num_episodes": 2000, "min_survival": 30},
    "walk_forward": {"num_episodes": 2000, "min_survival": 50},
    "raise_right_hand": {"num_episodes": 2000, "min_survival": 50},
    "wave_right_hand": {"num_episodes": 2000, "min_survival": 50},
}


def load_policy_and_normalize(skill_id: str):
    """Load a trained policy and its VecNormalize stats."""
    skill_dir = SKILLS_DIR / skill_id
    model_path = skill_dir / "model.zip"
    normalize_path = skill_dir / "vec_normalize.pkl"

    policy = PPO.load(str(model_path))

    vec_env = None
    if normalize_path.exists():
        # Create a dummy env to load VecNormalize into
        dummy_config = ExperimentConfig(
            skill_id=skill_id, name=skill_id, description="",
            reward_components=["upright_reward"],
            reward_weights={"upright_reward": 1.0},
        )
        dummy_env = DummyVecEnv([lambda: G1SkillEnv(dummy_config)])
        vec_env = VecNormalize.load(str(normalize_path), dummy_env)
        vec_env.training = False

    return policy, vec_env


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print("=" * 60)
    print("ATLAS Phase 3: Trajectory Collection")
    print("=" * 60)

    recorder = TrajectoryRecorder(
        output_dir=OUTPUT_DIR,
        image_size=128,
        camera_name="track",
    )

    stats = {}

    for skill_id in LANGUAGE_COMMANDS:
        print(f"\n{'='*40}")
        print(f"SKILL: {skill_id}")
        print(f"{'='*40}")

        if not (SKILLS_DIR / skill_id / "model.zip").exists():
            print(f"  SKIP: no trained model found")
            continue

        policy, vec_env = load_policy_and_normalize(skill_id)
        config = COLLECTION_CONFIG[skill_id]

        print(f"  Episodes: {config['num_episodes']}")
        print(f"  Min survival: {config['min_survival']} steps")
        print(f"  Language commands: {len(LANGUAGE_COMMANDS[skill_id])}")

        output_path = recorder.collect_trajectories(
            policy=policy,
            vec_normalize=vec_env,
            skill_id=skill_id,
            language_commands=LANGUAGE_COMMANDS[skill_id],
            num_episodes=config["num_episodes"],
            max_steps=500,
            min_survival_steps=config["min_survival"],
        )

        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        stats[skill_id] = {
            "file": str(output_path),
            "size_mb": round(size_mb, 1),
        }
        print(f"  Saved: {output_path.name} ({size_mb:.1f} MB)")

    elapsed = time.time() - start_time

    # Save stats
    stats["total_time_minutes"] = round(elapsed / 60, 1)
    stats_path = OUTPUT_DIR.parent / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for skill_id, info in stats.items():
        if isinstance(info, dict):
            print(f"  {skill_id:>20}: {info['size_mb']} MB")
    print(f"\nStats saved to: {stats_path}")

    recorder.close()


if __name__ == "__main__":
    main()
