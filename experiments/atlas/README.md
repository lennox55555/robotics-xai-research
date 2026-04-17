# ATLAS Experiment Plan
**Autonomous Transfer Learning to Action-conditioned Specialists**

## Context

The project trains specialist RL policies for a Unitree G1 humanoid robot via MuJoCo simulation. A deep audit revealed 8+ bugs (VecNormalize, dead code, unimplemented rewards, missing curriculum learning, no trajectory recording, no cameras). This plan fixes those bugs, trains 6 specialist policies, distills them into a language-conditioned VLA generalist, runs XAI comparison, and tests compositional generalization. The goal is a publishable paper.

---

## Hypotheses

**H1 (Primary):** A language-conditioned VLA distilled from 6 RL specialists will reproduce ≥85% of each specialist's mean episode reward while enabling compositional skill execution from novel language prompts.

**H0_1 (Null):** The VLA achieves <50% of average specialist reward (distillation fails).

**H2:** VLA saliency maps diverge significantly from specialist saliency (cosine similarity <0.7), with VLA attending broadly while specialists attend narrowly.

**H3:** The VLA achieves >0% success on held-out compositional prompts (e.g. "walk forward then raise your right hand") that no specialist was trained on.

---

## Plan Overview

| Phase | What | Duration |
|-------|------|----------|
| 1 | Bug fixes | 3-5 days |
| 2 | Train 6 specialist policies | 5-7 days |
| 3 | Collect trajectory demonstrations | 2-3 days |
| 4 | Distill into VLA (Octo, then OpenVLA) | 5-7 days |
| 5 | XAI comparison (saliency, SHAP, attention) | 4-5 days |
| 6 | Compositional generalization testing | 2-3 days |

---

## Phase 1: Bug Fixes

### BF1. Implement missing reward components
**File:** `src/experiments/experiment_runner.py` (after line 306 in `_get_reward()`)
- Add `elif` branches for: `foot_clearance`, `joint_limit_penalty`, `self_collision_penalty`
- These use `data.contact`, `data.qpos` vs `model.jnt_range`, and swing foot height

### BF2. Implement curriculum learning
**File:** `src/experiments/experiment_runner.py` (in `ExperimentRunner.run()`)
- Create a `CurriculumCallback(BaseCallback)` that reads `config.curriculum` stages
- At stage boundaries, update `config.target_velocity` (or target_height for jump)
- The env reads `self.config.target_velocity` in `_get_reward()` already — just need to mutate it

### BF3. Use termination conditions from config
**File:** `src/experiments/experiment_runner.py` `G1SkillEnv._is_terminated()`
- Parse `self.config` termination thresholds instead of hardcoded `< 0.4` and `< 0.3`
- Fallback to current hardcoded values if config doesn't specify

### BF4. Fix app.py VecNormalize handling
**File:** `app.py` — `load_skill()` (line 644) and `normalize_observation()` (line 692)
- Replace manual pickle extraction + manual normalization math with proper `VecNormalize.load()` wrapping a DummyVecEnv
- Store the loaded VecNormalize env in state, use `vec_env.normalize_obs(obs)` instead of manual math

### BF5. Fix app.py LiveViewCallback
**File:** `app.py` lines 80-125
- The reload at line 105 (`PPO.load()` without env) means live view uses unnormalized obs
- Fix: share the VecNormalize env from the training thread, or just use the manual normalization (which BF4 fixes)

### BF6. Add cameras to MuJoCo XML
**File:** `mujoco_menagerie/unitree_g1/scene_with_hands.xml` — add inside `<worldbody>`:
```xml
<camera name="track" pos="0 -3 1.5" xyaxes="1 0 0 0 0.4472 0.8944" mode="trackcom"/>
<camera name="front_fixed" pos="3 0 1.2" xyaxes="0 1 0 -0.3714 0 0.9285"/>
```
**File:** `mujoco_menagerie/unitree_g1/g1_with_hands.xml` — inside `torso_link` body (after line 193):
```xml
<camera name="ego" pos="0.1 0 0.3" xyaxes="0 -1 0 0.5 0 0.866" fovy="90"/>
```

### BF7. Build trajectory recorder
**New file:** `src/utils/trajectory_recorder.py`
- Records per-timestep: `observation`, `action`, `reward`, `image` (from track camera), `language_command`, `terminated`
- Saves to HDF5 (one file per skill, chunked by episode)
- Images at 128x128 with compression (~4GB total for full dataset)

### BF8. Add new skill templates
**File:** `src/robot/robot_spec.py` SKILL_TEMPLATES dict
- Add `turn_left` template (reward: upright + yaw_rotation + energy_efficiency + fall_penalty)
- Add `wave_right_hand` template (reward: right_hand_height + wave_motion + upright + smoothness)
**New files:** `skills/configs/turn_left.json`, `skills/configs/wave_right_hand.json`

---

## Phase 2: Train 6 Specialist Policies

| Skill | Timesteps | Transfer From | Success Criterion |
|-------|-----------|---------------|-------------------|
| balance_stand | 500k (continue from 100k) | None | Stand 500 steps, torso < 0.2 rad |
| walk_forward | 5M (continue from 2.2M) | balance_stand | Walk 5m, vel > 0.5 m/s |
| jump | 3M (fresh — corrupted) | balance_stand | Height > 0.1m, land stable |
| raise_right_hand | 2M (continue from 500k) | balance_stand | Hand > 1.5m for 100 steps |
| turn_left | 1M (new) | balance_stand | 90° turn, no fall |
| wave_right_hand | 1.5M (new) | raise_right_hand | Hand > 1.2m + lateral oscillation |

**Training order:** balance_stand → walk_forward / jump / raise_right_hand (parallel) → turn_left → wave_right_hand

**Gate:** Each specialist must pass 100 eval episodes with >80% survival rate before Phase 3.

---

## Phase 3: Trajectory Collection

For each trained specialist:
1. Load policy + VecNormalize stats
2. Roll out 5000 episodes (deterministic), keep ~500-1000 successful ones
3. Record per step: raw obs, normalized obs, action, reward, 128x128 RGB from `track` camera, language command
4. Language commands: 4 phrasings per skill, randomly sampled per episode for diversity
5. Save as HDF5: `experiments/atlas/phase3_trajectories/raw/{skill_id}_trajectories.hdf5`
6. Convert to RLDS format for Octo: `experiments/atlas/phase3_trajectories/rlds/`

**Storage estimate:** ~4GB compressed for 6 skills × 500 episodes × 500 steps × 128×128 images

---

## Phase 4: VLA Distillation

### 4a: Octo (primary — lighter, local inference possible)
- Fine-tune `octo-base` (93M params) from HuggingFace `rail-berkeley/octo-base`
- Freeze vision encoder + language encoder, train cross-attention + diffusion action head
- Remap from Octo's 7-DOF default to our 43-DOF via learned linear projection
- **Requires:** 1× A100 GPU, ~6-12 hours

### 4b: OpenVLA (secondary — bigger LLM backbone)
- LoRA fine-tune `openvla-7b` on same trajectory data
- Discretizes 43-DOF actions into 256 bins per dim
- **Requires:** 4× A100 GPU, ~24 hours

### Evaluation
- 100 episodes per skill per condition (VLA-Octo, VLA-OpenVLA, Specialist, Random)
- Measure: mean reward, success rate, survival steps, action smoothness

---

## Phase 5: XAI Comparison

### Methods (3 methods × 2 model types × 6 skills)

1. **Input Saliency** — extend existing `PolicyAnalyzer.compute_saliency()` to also run on VLA proprio pathway. 100 obs per skill.
2. **SHAP** — `shap.GradientExplainer` on both specialist MLP and VLA proprio head. Compare feature attribution rankings.
3. **Attention Maps** (VLA only) — extract cross-attention between language tokens and image patches. Visualize per-skill.

### Statistical Tests
- Paired Wilcoxon on saliency entropy (specialist vs VLA)
- Bootstrap 95% CI on cosine similarity
- Bonferroni correction for 6 skills

### Outputs
- Feature importance heatmap (6 skills × ~155 features, specialist vs VLA side by side)
- Joint-group attribution bar chart (which body parts does each model attend to)
- Saliency divergence boxplot
- Attention map overlays on camera images

---

## Phase 6: Compositional Generalization

Test VLA on 10 held-out prompts combining 2+ skills:

| # | Prompt | Skills Combined |
|---|--------|-----------------|
| 1 | "Walk forward then stop and balance" | walk + balance |
| 2 | "Walk forward and raise your right hand" | walk + raise |
| 3 | "Jump then wave" | jump + wave |
| 4 | "Turn left and walk forward" | turn + walk |
| 5 | "Raise your right hand while standing still" | balance + raise |
| 6 | "Wave while walking forward" | walk + wave |
| 7 | "Jump three times" | jump × 3 |
| 8 | "Walk forward, stop, and turn left" | walk + balance + turn |
| 9 | "Raise both hands" | raise right + implicit left |
| 10 | "Walk backward" (zero-shot) | novel — never trained |

50 episodes per prompt, 2000 steps max. Score sub-goals independently (partial credit) + overall composition success.

---

## Directory Structure

```
experiments/atlas/
  README.md
  config.yaml
  phase1_bugfixes/CHECKLIST.md
  phase2_specialists/
    configs/                    # Per-skill training configs
    training_logs/              # W&B run IDs, curves
    evaluation/                 # 100-episode eval results per skill
  phase3_trajectories/
    raw/                        # HDF5 files per skill
    rlds/                       # Converted for Octo/OpenVLA
    stats.json
  phase4_vla/
    octo/{config.yaml, checkpoints/, final_model/, training_logs/}
    openvla/{same structure}
    evaluation/                 # Per-skill VLA results
  phase5_xai/
    saliency/{specialist/, vla/}
    feature_importance/{specialist/, vla/}
    attention_maps/
    figures/                    # Publication-ready
    statistical_tests.json
  phase6_composition/
    prompts.json
    results/
    videos/
    analysis.json
  scripts/
    train_specialist.py
    collect_trajectories.py
    convert_hdf5_to_rlds.py
    finetune_octo.py
    evaluate_vla.py
    run_xai_comparison.py
    evaluate_composition.py
    generate_figures.py
```

---

## New Dependencies

```
h5py>=3.10.0               # Trajectory storage
tensorflow>=2.15.0          # RLDS data pipeline
tensorflow-datasets>=4.9.0
octo                        # pip install from github
jax[cpu]                    # Local inference (jax[cuda12] on GPU)
flax>=0.8.0
pingouin>=0.5.3             # Statistical tests
mediapy>=1.2.0              # Video display in notebooks
opencv-python>=4.8.0        # Image processing
```

---

## Verification Plan

1. **After Phase 1:** Run `python -c "from src.experiments.experiment_runner import ExperimentRunner; print('OK')"` — imports clean. Run a 10k-step balance_stand training to verify curriculum callback, reward components, and video recording all fire.
2. **After Phase 2:** Run eval script for each skill, verify >80% survival rate and reward above threshold.
3. **After Phase 3:** Open HDF5 files, verify image shapes (128×128×3), obs dimensions match, episode counts correct.
4. **After Phase 4:** Run VLA in MuJoCo loop, verify it produces 43-dim actions and the robot moves sensibly.
5. **After Phase 5:** Check that saliency arrays are non-zero, cosine similarities are computed, figures render.
6. **After Phase 6:** Review composition videos qualitatively, verify scoring logic.
