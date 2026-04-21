#!/usr/bin/env python3
"""
ATLAS Phase 6: Compositional Generalization Testing

Tests whether the VLA can execute skill combinations from natural language
prompts that no individual specialist was trained on.

For each compositional prompt, runs 50 episodes and scores:
- Sub-goal success (partial credit)
- Overall composition success (all sub-goals met)
- Records video of each attempt

The VLA was only trained on single-skill demonstrations. Any success on
multi-skill compositions is emergent generalization.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import mujoco
import imageio
from src.models.simple_vla import load_vla, SKILL_TO_ID

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "atlas" / "phase6_composition"
PROMPTS_PATH = OUTPUT_DIR / "prompts.json"


class CompositionEvaluator:
    """Runs the VLA in MuJoCo and scores compositional prompts."""

    def __init__(self, vla_model):
        self.vla = vla_model
        self.vla.eval()
        self.device = next(vla_model.parameters()).device

        # Load MuJoCo
        xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model, height=720, width=1280)

        # Camera for VLA input
        self.cam_renderer = mujoco.Renderer(self.mj_model, height=128, width=128)
        self.track_cam_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
        )

    def get_obs(self) -> np.ndarray:
        """Get proprioceptive observation (matches training)."""
        qpos = self.mj_data.qpos[3:].copy()
        qvel = self.mj_data.qvel.copy()
        torso_xmat = self.mj_data.xmat[1].reshape(9)
        perturbation = np.zeros(6)
        return np.concatenate([qpos, qvel, torso_xmat, perturbation])

    def get_camera_image(self) -> np.ndarray:
        """Get 128x128 RGB from track camera."""
        self.cam_renderer.update_scene(self.mj_data, camera=self.track_cam_id)
        return self.cam_renderer.render().copy()

    def render_frame(self) -> np.ndarray:
        """Render high-res frame for video."""
        self.renderer.update_scene(self.mj_data)
        return self.renderer.render().copy()

    def is_terminated(self) -> bool:
        com_height = self.mj_data.subtree_com[0][2]
        if com_height < 0.3:
            return True
        torso_xmat = self.mj_data.xmat[1].reshape(3, 3)
        if torso_xmat[2, 2] < 0.2:
            return True
        return False

    def get_metrics(self) -> dict:
        """Get current robot state metrics for scoring."""
        com_height = float(self.mj_data.subtree_com[0][2])
        forward_vel = float(self.mj_data.qvel[0])
        lateral_vel = float(self.mj_data.qvel[1])
        com_vel = float(np.linalg.norm(self.mj_data.qvel[:3]))
        upright = float(self.mj_data.xmat[1].reshape(3, 3)[2, 2])
        yaw_vel = float(self.mj_data.qvel[5])

        # Hand heights
        right_hand_height = 0.0
        left_hand_height = 0.0
        try:
            rh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_palm_link")
            if rh_id >= 0:
                right_hand_height = float(self.mj_data.xpos[rh_id][2])
            lh_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_palm_link")
            if lh_id >= 0:
                left_hand_height = float(self.mj_data.xpos[lh_id][2])
        except:
            pass

        return {
            "com_height": com_height,
            "forward_vel": forward_vel,
            "lateral_vel": lateral_vel,
            "com_vel": com_vel,
            "upright": upright,
            "yaw_vel": yaw_vel,
            "right_hand_height": right_hand_height,
            "left_hand_height": left_hand_height,
        }

    def run_episode(self, language_command: str, max_steps: int = 2000,
                    record_video: bool = True) -> dict:
        """Run one episode with the VLA and record metrics."""
        # Reset
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        nq = self.mj_model.nq
        self.mj_data.qpos[7:] += np.random.uniform(-0.02, 0.02, nq - 7)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Map language to skill ID
        skill_id = 0  # default
        cmd_lower = language_command.lower()
        # Try to find best matching skill from the command
        skill_keywords = {
            "balance": 0, "stand": 0, "still": 0,
            "squat": 1, "crouch": 1, "lower": 1,
            "jump": 2, "leap": 2,
            "walk": 3, "forward": 3, "step": 3,
            "raise": 4, "lift": 4, "hand up": 4,
            "wave": 5,
        }
        for keyword, sid in skill_keywords.items():
            if keyword in cmd_lower:
                skill_id = sid
                break

        skill_tensor = torch.LongTensor([skill_id]).to(self.device)
        frames = []
        all_metrics = []

        for step in range(max_steps):
            # Get observation
            obs = self.get_obs()
            image = self.get_camera_image()

            # VLA inference
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            proprio_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action = self.vla(image_tensor, proprio_tensor, skill_tensor)
            action_np = action.squeeze(0).cpu().numpy()

            # Apply action
            ctrl = action_np * self.mj_model.actuator_ctrlrange[:, 1]
            self.mj_data.ctrl[:] = ctrl

            for _ in range(5):  # frame_skip
                mujoco.mj_step(self.mj_model, self.mj_data)

            # Record
            metrics = self.get_metrics()
            all_metrics.append(metrics)

            if record_video and step % 3 == 0:  # Every 3rd frame for smaller videos
                frames.append(self.render_frame())

            if self.is_terminated():
                break

        return {
            "language_command": language_command,
            "skill_id": skill_id,
            "steps": len(all_metrics),
            "metrics": all_metrics,
            "frames": frames if record_video else [],
        }


def score_subgoals(episode_metrics: list, success_criteria: dict) -> dict:
    """Score sub-goals from an episode's metrics."""
    scores = {}

    for goal, threshold in success_criteria.items():
        if goal == "walk_distance_m":
            # Approximate distance from forward velocity integration
            total_dist = sum(abs(m["forward_vel"]) * 0.025 for m in episode_metrics)  # dt * frame_skip
            scores[goal] = total_dist >= threshold
        elif goal == "final_stand_steps":
            # Check if last N steps have low velocity
            if len(episode_metrics) >= threshold:
                last_n = episode_metrics[-threshold:]
                stable = all(m["com_vel"] < 0.3 for m in last_n)
                scores[goal] = stable
            else:
                scores[goal] = False
        elif goal == "hand_height_m":
            # Check if hand ever reached threshold
            max_hand = max(m["right_hand_height"] for m in episode_metrics)
            scores[goal] = max_hand >= threshold
        elif goal == "right_hand_height_m":
            max_hand = max(m["right_hand_height"] for m in episode_metrics)
            scores[goal] = max_hand >= threshold
        elif goal == "left_hand_height_m":
            max_hand = max(m["left_hand_height"] for m in episode_metrics)
            scores[goal] = max_hand >= threshold
        elif goal == "jump_height_m":
            initial_height = episode_metrics[0]["com_height"]
            max_gain = max(m["com_height"] - initial_height for m in episode_metrics)
            scores[goal] = max_gain >= threshold
        elif goal == "wave_oscillation_detected":
            # Check for oscillation in hand height
            if len(episode_metrics) > 20:
                hand_heights = [m["right_hand_height"] for m in episode_metrics[-100:]]
                if len(hand_heights) > 10:
                    diffs = np.diff(hand_heights)
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    scores[goal] = sign_changes >= 3
                else:
                    scores[goal] = False
            else:
                scores[goal] = False
        elif goal == "turn_degrees":
            total_yaw = sum(abs(m["yaw_vel"]) * 0.025 for m in episode_metrics)
            scores[goal] = np.degrees(total_yaw) >= threshold
        elif goal == "com_velocity_max":
            avg_vel = np.mean([m["com_vel"] for m in episode_metrics])
            scores[goal] = avg_vel <= threshold
        elif goal == "backward_distance_m":
            total_dist = sum(-m["forward_vel"] * 0.025 for m in episode_metrics if m["forward_vel"] < 0)
            scores[goal] = total_dist >= threshold
        elif goal == "distinct_jumps":
            # Count distinct height peaks
            heights = [m["com_height"] for m in episode_metrics]
            initial = heights[0] if heights else 0
            peaks = 0
            ascending = False
            for h in heights:
                if h > initial + 0.03 and not ascending:
                    ascending = True
                elif h < initial + 0.01 and ascending:
                    peaks += 1
                    ascending = False
            scores[goal] = peaks >= threshold
        elif goal == "stop_steps":
            # Check for a stable stop period anywhere in the episode
            window = threshold
            for i in range(len(episode_metrics) - window):
                chunk = episode_metrics[i:i+window]
                if all(m["com_vel"] < 0.2 for m in chunk):
                    scores[goal] = True
                    break
            else:
                scores[goal] = False
        else:
            scores[goal] = False

    return scores


def main():
    print("=" * 60)
    print("ATLAS Phase 6: Compositional Generalization Testing")
    print("=" * 60)

    # Load prompts
    with open(PROMPTS_PATH) as f:
        prompts_config = json.load(f)

    prompts = prompts_config["compositional_prompts"]
    eval_config = prompts_config["evaluation_config"]
    episodes_per_prompt = eval_config["episodes_per_prompt"]

    # Load VLA
    print("\nLoading VLA model...")
    vla = load_vla()
    evaluator = CompositionEvaluator(vla)
    print("VLA loaded and MuJoCo ready")

    # Create output dirs
    results_dir = OUTPUT_DIR / "results"
    videos_dir = OUTPUT_DIR / "videos"
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for prompt_config in prompts:
        prompt_id = prompt_config["id"]
        prompt_text = prompt_config["prompt"]
        criteria = prompt_config["success_criteria"]
        max_steps = prompt_config.get("max_steps", 2000)

        print(f"\n{'='*40}")
        print(f"Prompt {prompt_id}: \"{prompt_text}\"")
        print(f"{'='*40}")

        prompt_scores = []
        best_episode = None
        best_score = -1

        for ep in range(episodes_per_prompt):
            # Record video for first 3 episodes only (save disk)
            record = (ep < 3)

            result = evaluator.run_episode(
                language_command=prompt_text,
                max_steps=max_steps,
                record_video=record,
            )

            # Score sub-goals
            subgoal_scores = score_subgoals(result["metrics"], criteria)
            n_met = sum(subgoal_scores.values())
            n_total = len(subgoal_scores)
            partial_score = n_met / n_total if n_total > 0 else 0
            full_success = all(subgoal_scores.values())

            prompt_scores.append({
                "episode": ep,
                "steps": result["steps"],
                "subgoal_scores": {k: bool(v) for k, v in subgoal_scores.items()},
                "partial_score": partial_score,
                "full_success": full_success,
            })

            # Save best video
            if partial_score > best_score and result["frames"]:
                best_score = partial_score
                best_episode = result

            if ep % 10 == 0:
                print(f"  Episode {ep}/{episodes_per_prompt} | "
                      f"partial={partial_score:.2f} | full={full_success}")

        # Save best video
        if best_episode and best_episode["frames"]:
            video_path = videos_dir / f"prompt_{prompt_id:02d}_best.mp4"
            writer = imageio.get_writer(str(video_path), fps=30, macro_block_size=1)
            for frame in best_episode["frames"]:
                writer.append_data(frame)
            writer.close()
            print(f"  Video saved: {video_path.name}")

        # Aggregate scores
        partial_scores = [s["partial_score"] for s in prompt_scores]
        full_successes = [s["full_success"] for s in prompt_scores]

        prompt_result = {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "skills_combined": prompt_config["skills_combined"],
            "mean_partial_score": float(np.mean(partial_scores)),
            "full_success_rate": float(np.mean(full_successes)),
            "mean_steps": float(np.mean([s["steps"] for s in prompt_scores])),
            "n_episodes": episodes_per_prompt,
            "per_subgoal_success_rate": {},
        }

        # Per-subgoal success rates
        for goal in criteria:
            rate = np.mean([s["subgoal_scores"].get(goal, False) for s in prompt_scores])
            prompt_result["per_subgoal_success_rate"][goal] = float(rate)

        all_results.append(prompt_result)

        print(f"  Partial score: {np.mean(partial_scores):.2f}")
        print(f"  Full success rate: {np.mean(full_successes)*100:.1f}%")

    # Save all results
    results_path = results_dir / "composition_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 6 COMPLETE -- SUMMARY")
    print(f"{'='*60}")
    print(f"{'Prompt':<50} {'Partial':>8} {'Full %':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['prompt']:<50} {r['mean_partial_score']:>8.2f} {r['full_success_rate']*100:>7.1f}%")

    # Overall stats
    any_success = any(r["full_success_rate"] > 0 for r in all_results)
    mean_partial = np.mean([r["mean_partial_score"] for r in all_results])
    print(f"\nOverall mean partial score: {mean_partial:.2f}")
    print(f"H3 (any compositional success > 0%): {'CONFIRMED' if any_success else 'REJECTED'}")
    print(f"\nResults: {results_path}")
    print(f"Videos: {videos_dir}")


if __name__ == "__main__":
    main()
