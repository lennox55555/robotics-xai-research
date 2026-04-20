# ATLAS Phase 2: Specialist Training Log

## balance_stand

### Run 1: Fixed perturbation (5M steps, 2026-04-17)
- **Config:** 43-DOF model, fixed 30N pushes at 10% of steps, obs_dim=105
- **Network:** [512, 512], LR=3e-4, max_episode_steps=1000
- **Result:** Reward 86 -> 226, episode length 46 -> 106 steps
- **Plateau:** Around 800k steps, stayed in 210-230 range
- **Issue:** Fixed perturbation too easy once learned; no force sensing in obs
- **Videos:** 52 milestone recordings in experiments/runs/balance_stand_20260417_210132/videos/

### Run 2: Progressive perturbation + force sensing (5M steps, 2026-04-17)
- **Network:** [512, 512], LR=3e-4, max_episode_steps=1000
- **Changes from Run 1:**
  1. Added perturbation force/torque to observation space (obs_dim 105 -> 111)
     - 6 new dims: normalized force_xyz + torque_xyz in [-1, 1] range
     - Lets the policy "feel" the push and counter proactively
  2. Progressive perturbation curriculum:
     - Force: 10N (start) -> 80N (end), linear ramp over total_timesteps
     - Torque: 2Nm (start) -> 15Nm (end)
     - Probability: 10% of steps (unchanged)
  3. Fresh training (old checkpoint incompatible due to obs dim change)
- **Result:** Reward 84 -> 200, episode length 45 -> 95 steps (peak 208/99)
- **Plateau:** Around 500k steps at ~180-200 range
- **Analysis:** Slightly lower than Run 1 final numbers, but facing harder forces
  at end of training (ramp reached ~28N at last logged step). Force sensing
  added but may need more capacity to exploit it.

### Run 3: Deeper network + lower LR + longer episodes (5M steps, 2026-04-17)
- **Network:** [512, 256, 128] (3 layers, more capacity for feature extraction)
- **LR:** 1e-4 (reduced from 3e-4 for finer optimization)
- **max_episode_steps:** 2000 (doubled from 1000, more practice time per episode)
- **Progressive perturbation:** Same as Run 2 (10N->80N force, 2Nm->15Nm torque)
- **obs_dim:** 111 (same as Run 2, includes force sensing)
- **Fresh training:** Required due to network architecture change
- **Rationale:**
  - Deeper network gives more layers to process the 111-dim obs (especially
    the perturbation force channels) and extract better features
  - Lower LR prevents overshooting -- the policy needs fine-grained corrections
    to counter specific force vectors, not large weight updates
  - Longer episodes let the robot practice sustained balance and recovery from
    multiple pushes in sequence, not just one push before timeout
- **Result:** Reward 78 -> 231, episode length 42 -> 110 steps
- **Peak:** 231.3 / 109.7 at step 1.14M
- **Improvement over Run 2:** +11% reward, +10% ep length (while facing harder forces)

### Run 3 continuation: Same config, +5M steps (10M cumulative, 2026-04-17)
- **Resumed from:** Run 3 checkpoint at 5M steps
- **Config:** Same as Run 3 ([512, 256, 128], LR=1e-4, max_ep=2000)
- **Result:** Reward 284, episode length 133 steps
- **Peak:** 297.9 / 139.3 at step 6.19M cumulative
- **Analysis:** +23% reward over Run 3 end. Robot survives ~14 seconds against
  progressive pushes. Plateauing in 280-290 range at end -- ready for LR reduction.

### Run 3 fine-tune: LR 3e-5, +5M steps (15M cumulative, 2026-04-17)
- **Change:** Learning rate reduced 1e-4 -> 3e-5 for fine-grained optimization
- **Resumed from:** 10M checkpoint (reward ~284)
- **Rationale:** Policy has good overall behavior but reward plateaued at 280-290.
  Lower LR allows smaller weight updates to polish the policy without
  disrupting learned recovery patterns. Like sanding with finer grit.
- **Result:** TBD

## jump
- **Status:** Not started (waiting for balance_stand to be solid)

## walk_forward
- **Status:** Not started

## raise_right_hand
- **Status:** Not started

## squat
- **Status:** Not started

## wave_right_hand
- **Status:** Not started
