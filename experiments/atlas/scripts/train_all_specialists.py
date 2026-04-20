#!/usr/bin/env python3
"""
ATLAS Phase 2: Automated Specialist Training Pipeline

Trains all 6 specialist skills in dependency order with automatic
continuation if a skill hasn't reached its reward threshold.

Training order:
  balance_stand (already done, 15M steps)
  └── squat (running or just finished)
      └── jump (transfers from squat)
  └── walk_forward (transfers from balance_stand)
  └── raise_right_hand (transfers from balance_stand)
      └── wave_right_hand (transfers from raise_right_hand)

Each skill trains for an initial batch of steps, then checks if it
meets its reward threshold. If not, it continues with a lower LR.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.experiment_runner import train_skill, ExperimentRunner

SKILLS_DIR = PROJECT_ROOT / "skills" / "trained"
LOG_FILE = PROJECT_ROOT / "experiments" / "atlas" / "phase2_specialists" / "training_logs" / "automated_training.log"


def log(msg: str):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def get_last_reward(skill_id: str) -> float:
    """Get the last logged reward for a skill from its most recent run."""
    runs_dir = PROJECT_ROOT / "experiments" / "runs"
    skill_runs = sorted(runs_dir.glob(f"{skill_id}_*"), reverse=True)
    for run_dir in skill_runs:
        progress = run_dir / "progress.csv"
        if progress.exists():
            lines = progress.read_text().strip().split("\n")
            if lines:
                last = lines[-1].split(",")
                return float(last[1])
    return 0.0


def skill_exists(skill_id: str) -> bool:
    """Check if a trained model exists for this skill."""
    return (SKILLS_DIR / skill_id / "model.zip").exists()


def train_with_threshold(
    skill_id: str,
    initial_steps: int,
    reward_threshold: float,
    max_total_steps: int,
    hidden_sizes: list = None,
    initial_lr: float = 1e-4,
    finetune_lr: float = 3e-5,
    max_episode_steps: int = 2000,
):
    """Train a skill until it reaches a reward threshold or max steps.

    Phase 1: Train at initial_lr for initial_steps
    Phase 2: If below threshold, fine-tune at finetune_lr for another batch
    Phase 3: Repeat until threshold met or max_total_steps reached
    """
    if hidden_sizes is None:
        hidden_sizes = [512, 256, 128]

    total_trained = 0

    # Phase 1: Initial training
    log(f"  Phase 1: Training {skill_id} for {initial_steps:,} steps at LR={initial_lr}")
    result = train_skill(
        skill_id,
        timesteps=initial_steps,
        hidden_sizes=hidden_sizes,
        learning_rate=initial_lr,
        max_episode_steps=max_episode_steps,
    )
    total_trained += initial_steps

    reward = get_last_reward(skill_id)
    log(f"  Phase 1 result: reward={reward:.1f} (threshold={reward_threshold:.1f})")

    # Phase 2+: Fine-tune if below threshold
    while reward < reward_threshold and total_trained < max_total_steps:
        remaining = min(initial_steps, max_total_steps - total_trained)
        if remaining <= 0:
            break
        log(f"  Below threshold. Fine-tuning {skill_id} for {remaining:,} more steps at LR={finetune_lr}")
        result = train_skill(
            skill_id,
            timesteps=remaining,
            hidden_sizes=hidden_sizes,
            learning_rate=finetune_lr,
            max_episode_steps=max_episode_steps,
        )
        total_trained += remaining
        reward = get_last_reward(skill_id)
        log(f"  Fine-tune result: reward={reward:.1f} (threshold={reward_threshold:.1f})")

    if reward >= reward_threshold:
        log(f"  PASSED: {skill_id} reached {reward:.1f} >= {reward_threshold:.1f} in {total_trained:,} steps")
    else:
        log(f"  CAPPED: {skill_id} at {reward:.1f} after {total_trained:,} steps (threshold was {reward_threshold:.1f})")

    return reward, total_trained


def main():
    log("=" * 60)
    log("ATLAS Phase 2: Automated Specialist Training")
    log("=" * 60)
    start_time = time.time()

    # Skill training configs: (skill_id, initial_steps, reward_threshold, max_total_steps)
    # Thresholds are set conservatively -- we need "good enough" not perfect.
    training_plan = [
        # squat might already be trained or in progress -- check first
        {
            "skill_id": "squat",
            "initial_steps": 5_000_000,
            "reward_threshold": 300.0,
            "max_total_steps": 15_000_000,
        },
        {
            "skill_id": "jump",
            "initial_steps": 5_000_000,
            "reward_threshold": 200.0,
            "max_total_steps": 15_000_000,
        },
        {
            "skill_id": "walk_forward",
            "initial_steps": 5_000_000,
            "reward_threshold": 200.0,
            "max_total_steps": 15_000_000,
        },
        {
            "skill_id": "raise_right_hand",
            "initial_steps": 5_000_000,
            "reward_threshold": 200.0,
            "max_total_steps": 15_000_000,
        },
        {
            "skill_id": "wave_right_hand",
            "initial_steps": 5_000_000,
            "reward_threshold": 200.0,
            "max_total_steps": 15_000_000,
        },
    ]

    results = {}

    for config in training_plan:
        skill_id = config["skill_id"]
        log("")
        log(f"{'='*40}")
        log(f"SKILL: {skill_id}")
        log(f"{'='*40}")

        # Check if skill already has a model with good enough reward
        if skill_exists(skill_id):
            existing_reward = get_last_reward(skill_id)
            if existing_reward >= config["reward_threshold"]:
                log(f"  SKIP: {skill_id} already trained (reward={existing_reward:.1f} >= {config['reward_threshold']:.1f})")
                results[skill_id] = {"reward": existing_reward, "steps": 0, "status": "skipped"}
                continue
            else:
                log(f"  EXISTS but below threshold (reward={existing_reward:.1f}). Continuing training.")

        # Check prerequisites
        runner = ExperimentRunner()
        skill_config = runner.create_config_from_template(skill_id)
        if skill_config.transfer_from and not skill_exists(skill_config.transfer_from):
            log(f"  ERROR: prerequisite '{skill_config.transfer_from}' not trained. Skipping {skill_id}.")
            results[skill_id] = {"reward": 0, "steps": 0, "status": "missing_prereq"}
            continue

        reward, total_steps = train_with_threshold(
            skill_id=skill_id,
            initial_steps=config["initial_steps"],
            reward_threshold=config["reward_threshold"],
            max_total_steps=config["max_total_steps"],
        )
        results[skill_id] = {"reward": reward, "steps": total_steps, "status": "trained"}

    # Summary
    elapsed = time.time() - start_time
    log("")
    log("=" * 60)
    log("TRAINING COMPLETE")
    log(f"Total time: {elapsed/3600:.1f} hours")
    log("=" * 60)
    for skill_id, info in results.items():
        log(f"  {skill_id:>20}: reward={info['reward']:.1f} | steps={info['steps']:,} | {info['status']}")

    # Save results
    results_path = PROJECT_ROOT / "experiments" / "atlas" / "phase2_specialists" / "evaluation" / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
