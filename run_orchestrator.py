#!/usr/bin/env python3
"""
Multi-Agent Robot Skill Learning System

A sophisticated LLM-orchestrated system for teaching humanoid robots new skills.

Usage:
    python run_orchestrator.py              # Interactive mode
    python run_orchestrator.py --task "..." # Single task
    python run_orchestrator.py --train SKILL # Train a skill directly
    python run_orchestrator.py --list       # List available skills
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def interactive_mode():
    """Run in interactive chat mode."""
    from src.agents.orchestrator_v2 import create_orchestrator
    orchestrator = create_orchestrator()
    orchestrator.run_interactive()


def single_task_mode(task: str):
    """Run a single task and exit."""
    from src.agents.orchestrator_v2 import create_orchestrator

    print(f"Processing: {task}\n")
    print("-" * 60)

    orchestrator = create_orchestrator()
    response = orchestrator.process(task)
    print(f"\nAssistant: {response}\n")


def train_skill(skill_id: str, timesteps: int = None, render: bool = False):
    """Train a specific skill directly."""
    from src.experiments.experiment_runner import train_skill as do_train

    print(f"Training skill: {skill_id}")
    if timesteps:
        print(f"Timesteps: {timesteps:,}")
    if render:
        print("Rendering: Enabled (slower training)")

    result = do_train(skill_id, timesteps, render=render)
    print(f"\nTraining complete!")
    print(f"Result: {result}")


def evaluate_skill(skill_id: str, render: bool = False):
    """Evaluate a trained skill."""
    from src.experiments.experiment_runner import evaluate_skill as do_eval

    print(f"Evaluating skill: {skill_id}")
    result = do_eval(skill_id, n_episodes=10, render=render)
    return result


def list_skills():
    """List available skills."""
    from src.robot.robot_spec import SKILL_TEMPLATES

    print("\nAvailable Skill Templates:\n")
    print("-" * 60)

    for skill_id, template in SKILL_TEMPLATES.items():
        prereqs = template.get("prerequisites", [])
        prereq_str = f" (requires: {', '.join(prereqs)})" if prereqs else ""

        print(f"  {skill_id}")
        print(f"    {template['description']}{prereq_str}")
        print(f"    Timesteps: {template['training_config']['total_timesteps']:,}")
        print()

    print("-" * 60)
    print("\nTo train a skill:")
    print("  python run_orchestrator.py --train balance_stand")
    print("\nTo evaluate a trained skill:")
    print("  python run_orchestrator.py --eval walk_forward --render")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Robot Skill Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_orchestrator.py
      Start interactive mode

  python run_orchestrator.py --task "teach the robot to walk forward"
      Run a single task

  python run_orchestrator.py --list
      List available skill templates

  python run_orchestrator.py --train balance_stand
      Train the balance_stand skill

  python run_orchestrator.py --train walk_forward --timesteps 100000
      Train with custom timesteps

  python run_orchestrator.py --eval walk_forward --render
      Evaluate and visualize a trained skill
"""
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task instead of interactive mode"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available skill templates"
    )

    parser.add_argument(
        "--train",
        type=str,
        metavar="SKILL_ID",
        help="Train a specific skill"
    )

    parser.add_argument(
        "--eval",
        type=str,
        metavar="SKILL_ID",
        help="Evaluate a trained skill"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        help="Override training timesteps"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render visualization during evaluation"
    )

    args = parser.parse_args()

    if args.list:
        list_skills()
    elif args.train:
        train_skill(args.train, args.timesteps, args.render)
    elif args.eval:
        evaluate_skill(args.eval, args.render)
    elif args.task:
        single_task_mode(args.task)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
