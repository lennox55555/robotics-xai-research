#!/usr/bin/env python3
"""
View the G1 robot in MuJoCo viewer.

Usage:
    # View robot with trained model
    mjpython view_robot.py --skill balance_stand

    # View robot doing random actions
    mjpython view_robot.py --random

    # Just view the robot standing
    mjpython view_robot.py

NOTE: On macOS, run with `mjpython` instead of `python` for the viewer to work.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import mujoco.viewer


def load_model(skill_id: str = None):
    """Load a trained model if available."""
    if skill_id:
        model_path = PROJECT_ROOT / "skills" / "trained" / skill_id / "model.zip"
        if model_path.exists():
            from stable_baselines3 import PPO
            print(f"Loading trained model: {skill_id}")
            return PPO.load(str(model_path))
        else:
            print(f"No trained model found for: {skill_id}")
            return None
    return None


def get_observation(model, data):
    """Get observation vector (simplified)."""
    qpos = data.qpos[3:].copy()  # Skip root xyz
    qvel = data.qvel.copy()
    torso_xmat = data.xmat[1].reshape(9)
    return np.concatenate([qpos, qvel, torso_xmat])


def main():
    parser = argparse.ArgumentParser(description="View G1 robot in MuJoCo")
    parser.add_argument("--skill", type=str, help="Skill to load and execute")
    parser.add_argument("--random", action="store_true", help="Apply random actions")
    parser.add_argument("--steps", type=int, default=1000, help="Steps to run")

    args = parser.parse_args()

    # Load MuJoCo model
    xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(mj_model)

    # Load trained policy if specified
    policy = load_model(args.skill) if args.skill else None

    print("\nG1 Robot Viewer")
    print("=" * 40)
    print(f"DOF: {mj_model.nv}")
    print(f"Actuators: {mj_model.nu}")
    if args.skill:
        print(f"Skill: {args.skill}")
        if policy:
            print("Policy: Loaded")
        else:
            print("Policy: Not found (using zero actions)")
    elif args.random:
        print("Mode: Random actions")
    else:
        print("Mode: Standing (zero actions)")
    print("=" * 40)
    print("\nPress ESC in viewer to exit\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        step = 0
        while viewer.is_running() and step < args.steps:
            # Get action
            if policy:
                obs = get_observation(mj_model, mj_data)
                action, _ = policy.predict(obs, deterministic=True)
                mj_data.ctrl[:] = action * mj_model.actuator_ctrlrange[:, 1]
            elif args.random:
                # Random actions within range
                low = mj_model.actuator_ctrlrange[:, 0]
                high = mj_model.actuator_ctrlrange[:, 1]
                mj_data.ctrl[:] = np.random.uniform(low, high) * 0.1  # Small random
            else:
                # Zero actions (standing)
                mj_data.ctrl[:] = 0

            # Step simulation
            mujoco.mj_step(mj_model, mj_data)

            # Sync viewer
            viewer.sync()

            # Control speed
            time.sleep(0.01)

            step += 1

            # Print status occasionally
            if step % 100 == 0:
                com_height = mj_data.subtree_com[0][2]
                print(f"Step {step}: COM height = {com_height:.3f}m")

    print("Viewer closed.")


if __name__ == "__main__":
    main()
