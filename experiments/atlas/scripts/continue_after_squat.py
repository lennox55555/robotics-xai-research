#!/usr/bin/env python3
"""
Waits for the current squat training to finish (checks for model file),
then runs the remaining specialist training pipeline.

Run this in the background and go to sleep.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SKILLS_DIR = PROJECT_ROOT / "skills" / "trained"


def wait_for_squat():
    """Wait until squat model exists and is recent (training just finished)."""
    squat_model = SKILLS_DIR / "squat" / "model.zip"
    print("Waiting for current squat training to finish...")
    while True:
        if squat_model.exists():
            # Check if the file was modified in the last 5 minutes (just saved)
            import os
            mtime = os.path.getmtime(squat_model)
            if time.time() - mtime < 300:
                print(f"Squat model found (modified {time.time() - mtime:.0f}s ago). Proceeding.")
                return
            # Model exists but is old -- squat is still running or already done
            # Check if there's a running progress file being updated
            runs_dir = PROJECT_ROOT / "experiments" / "runs"
            squat_runs = sorted(runs_dir.glob("squat_*"), reverse=True)
            if squat_runs:
                progress = squat_runs[0] / "progress.csv"
                if progress.exists():
                    pmtime = os.path.getmtime(progress)
                    if time.time() - pmtime > 120:
                        # Progress file not updated in 2 min -- training is done
                        print("Squat training appears complete. Proceeding.")
                        return
        time.sleep(30)


def main():
    # Wait for squat
    wait_for_squat()
    time.sleep(5)  # Brief pause

    # Now run the full pipeline
    from train_all_specialists import main as run_pipeline
    run_pipeline()


if __name__ == "__main__":
    main()
