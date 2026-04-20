"""
Video Recorder for Training Milestones

Records evaluation episodes as mp4 videos at significant training milestones.
Integrates as a Stable-Baselines3 callback so it hooks directly into the
training loop without any external coordination.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class VideoRecorderCallback(BaseCallback):
    """
    Records a video of the policy running one full episode at milestone steps.

    Videos are saved to the experiment's videos/ directory with the step count
    in the filename, e.g. videos/episode_050000.mp4

    Uses a separate MuJoCo model/data instance for rendering so it doesn't
    interfere with the training environments.
    """

    def __init__(
        self,
        video_dir: Path,
        record_freq: int = 100_000,
        max_episode_steps: int = 1000,
        max_video_seconds: float = 5.0,
        video_fps: int = 30,
        render_width: int = 1280,
        render_height: int = 720,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.video_dir = Path(video_dir)
        self.record_freq = record_freq
        self.max_episode_steps = max_episode_steps
        self.max_frames = int(max_video_seconds * video_fps)
        self.video_fps = video_fps
        self.render_width = render_width
        self.render_height = render_height

        # Lazily initialized on first recording
        self._mj_model = None
        self._mj_data = None
        self._renderer = None
        self._last_recorded_step = -1

    def _init_renderer(self):
        """Initialize a dedicated MuJoCo model and renderer for video capture."""
        if self._renderer is not None:
            return

        xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._renderer = mujoco.Renderer(
            self._mj_model,
            height=self.render_height,
            width=self.render_width,
        )
        if self.verbose:
            print(f"[VideoRecorder] Renderer initialized ({self.render_width}x{self.render_height})")

    def _get_obs(self) -> np.ndarray:
        """Get observation from the dedicated render model (mirrors G1SkillEnv._get_obs)."""
        qpos = self._mj_data.qpos[3:].copy()
        qvel = self._mj_data.qvel.copy()
        torso_xmat = self._mj_data.xmat[1].reshape(9)
        # Zero perturbation during video recording (no pushes during eval)
        perturbation_obs = np.zeros(6)
        return np.concatenate([qpos, qvel, torso_xmat, perturbation_obs])

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize observation normalization if the training env uses it."""
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            return env.normalize_obs(obs)
        return obs

    def _record_episode(self, step_count: int):
        """Run one full episode with the current policy and save as video."""
        self._init_renderer()
        self.video_dir.mkdir(parents=True, exist_ok=True)

        video_path = self.video_dir / f"episode_{step_count:08d}.mp4"

        # Reset the render environment
        mujoco.mj_resetData(self._mj_model, self._mj_data)
        # Small random perturbation like training env
        nq = self._mj_model.nq
        self._mj_data.qpos[7:] += np.random.uniform(-0.02, 0.02, nq - 7)
        mujoco.mj_forward(self._mj_model, self._mj_data)

        frames = []
        frame_skip = 5  # Match G1SkillEnv frame_skip
        max_steps = min(self.max_episode_steps, self.max_frames)

        for step in range(max_steps):
            # Get observation and normalize it the same way training does
            raw_obs = self._get_obs()
            obs = self._normalize_obs(raw_obs)

            # Query policy
            action, _ = self.model.predict(obs, deterministic=True)

            # Apply action
            ctrl = action * self._mj_model.actuator_ctrlrange[:, 1]
            self._mj_data.ctrl[:] = ctrl

            for _ in range(frame_skip):
                mujoco.mj_step(self._mj_model, self._mj_data)

            # Render frame
            self._renderer.update_scene(self._mj_data)
            frame = self._renderer.render()
            frames.append(frame.copy())

            # Check termination (same as G1SkillEnv)
            com_height = self._mj_data.subtree_com[0][2]
            if com_height < 0.4:
                break
            torso_xmat = self._mj_data.xmat[1].reshape(3, 3)
            if torso_xmat[2, 2] < 0.3:
                break

        # Write video
        if frames:
            writer = imageio.get_writer(
                str(video_path),
                fps=self.video_fps,
                macro_block_size=1,
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            if self.verbose:
                print(f"[VideoRecorder] Saved {len(frames)} frames to {video_path} "
                      f"(step {step_count:,})")

        return video_path

    def _on_step(self) -> bool:
        """Check if we should record at this step."""
        current_step = self.num_timesteps

        # Record at step 0 (before any training), then at each record_freq
        should_record = False
        if self._last_recorded_step == -1 and current_step >= 0:
            should_record = True
        elif current_step - self._last_recorded_step >= self.record_freq:
            should_record = True

        if should_record:
            self._last_recorded_step = current_step
            try:
                self._record_episode(current_step)
            except Exception as e:
                if self.verbose:
                    print(f"[VideoRecorder] Recording failed at step {current_step}: {e}")

        return True

    def _on_training_end(self):
        """Record a final video when training completes."""
        try:
            self._record_episode(self.num_timesteps)
        except Exception as e:
            if self.verbose:
                print(f"[VideoRecorder] Final recording failed: {e}")

        # Clean up renderer
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
