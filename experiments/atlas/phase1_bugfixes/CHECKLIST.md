# Phase 1: Bug Fix Checklist

## Critical (Blocks Experiment)
- [x] BF1. Implement missing reward components (foot_clearance, joint_limit_penalty, self_collision_penalty, yaw_rotation)
- [x] BF2. Implement curriculum learning callback (CurriculumCallback)
- [x] BF3. Use termination conditions from skill config (min_com_height, min_upright parsed from JSON)
- [x] BF6. Add cameras to MuJoCo XML (track, front_fixed, ego) — all 3 verified rendering
- [x] BF7. Build trajectory recorder (HDF5 output with images, obs, actions, language)
- [x] BF8. Add turn_left and wave_right_hand skill templates + configs + yaw_rotation reward

## High (Affects Quality)
- [x] BF4. Fix app.py load_skill() — now uses VecNormalize.load() properly
- [x] BF5. Fix app.py LiveViewCallback — now calls load_skill() for proper normalization

## Verification
- [x] All imports clean (experiment_runner, video_recorder, trajectory_recorder, robot_spec)
- [x] 3 cameras render correctly (track 128x128, front_fixed, ego)
- [x] Termination thresholds parsed from config (walk=0.5, jump=0.3, balance upright=0.54)
- [x] Curriculum stages parsed (walk_forward: 3 velocity stages)
- [x] New skills verified (turn_left with yaw_rotation, wave_right_hand transferring from raise_right_hand)
- [ ] Run 10k-step balance_stand training — no errors (manual test)
