"""
Trajectory Recorder for VLA Distillation

Records full episode trajectories (observations, actions, rewards, images,
language commands) from trained specialist policies. Output is HDF5 format
suitable for conversion to RLDS and fine-tuning VLAs like Octo or OpenVLA.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import mujoco
from PIL import Image


class TrajectoryRecorder:
    """Records specialist policy rollouts as demonstration trajectories.

    Each trajectory includes per-timestep:
    - observation (proprioceptive state vector)
    - action (43-dim continuous)
    - reward (scalar)
    - image (RGB from track camera)
    - terminated/truncated flags
    - language command (string, per-episode)

    Stored as HDF5 with one group per episode.
    """

    def __init__(
        self,
        output_dir: Path,
        image_size: int = 128,
        camera_name: str = "track",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.camera_name = camera_name

        # Lazily initialized
        self._mj_model = None
        self._mj_data = None
        self._renderer = None

    def _init_mujoco(self):
        """Initialize a dedicated MuJoCo model and renderer."""
        if self._mj_model is not None:
            return

        xml_path = PROJECT_ROOT / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
        self._mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._mj_data = mujoco.MjData(self._mj_model)
        self._renderer = mujoco.Renderer(
            self._mj_model,
            height=self.image_size,
            width=self.image_size,
        )

    def _get_obs(self) -> np.ndarray:
        """Get proprioceptive observation (mirrors G1SkillEnv._get_obs)."""
        qpos = self._mj_data.qpos[3:].copy()
        qvel = self._mj_data.qvel.copy()
        torso_xmat = self._mj_data.xmat[1].reshape(9)
        return np.concatenate([qpos, qvel, torso_xmat])

    def _render_image(self) -> np.ndarray:
        """Render RGB image from the specified camera."""
        camera_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        self._renderer.update_scene(self._mj_data, camera=camera_id)
        return self._renderer.render().copy()

    def _is_terminated(self, min_com_height: float = 0.4, min_upright: float = 0.3) -> bool:
        """Check if episode should terminate."""
        com_height = self._mj_data.subtree_com[0][2]
        if com_height < min_com_height:
            return True
        torso_xmat = self._mj_data.xmat[1].reshape(3, 3)
        if torso_xmat[2, 2] < min_upright:
            return True
        return False

    def collect_trajectories(
        self,
        policy,
        vec_normalize,
        skill_id: str,
        language_commands: List[str],
        num_episodes: int = 5000,
        max_steps: int = 1000,
        min_survival_steps: int = 200,
        frame_skip: int = 5,
    ) -> Path:
        """Collect trajectory demonstrations from a trained specialist.

        Args:
            policy: Trained SB3 policy (PPO model)
            vec_normalize: VecNormalize env for observation normalization (or None)
            skill_id: Skill identifier
            language_commands: List of language phrasings for this skill
            num_episodes: Total episodes to attempt
            max_steps: Maximum steps per episode
            min_survival_steps: Minimum steps for a trajectory to be kept
            frame_skip: Physics substeps per action (match training env)

        Returns:
            Path to the saved HDF5 file.
        """
        self._init_mujoco()

        output_path = self.output_dir / f"{skill_id}_trajectories.hdf5"

        kept = 0
        attempted = 0

        with h5py.File(str(output_path), "w") as hf:
            # Metadata group
            meta = hf.create_group("metadata")
            meta.attrs["skill_id"] = skill_id
            meta.attrs["image_size"] = self.image_size
            meta.attrs["camera_name"] = self.camera_name
            meta.attrs["frame_skip"] = frame_skip
            meta.attrs["max_steps"] = max_steps

            for ep_idx in range(num_episodes):
                attempted += 1

                # Reset
                mujoco.mj_resetData(self._mj_model, self._mj_data)
                nq = self._mj_model.nq
                self._mj_data.qpos[7:] += np.random.uniform(-0.02, 0.02, nq - 7)
                mujoco.mj_forward(self._mj_model, self._mj_data)

                # Pick a random language command for this episode
                lang_cmd = language_commands[ep_idx % len(language_commands)]

                # Collect trajectory
                observations = []
                actions = []
                rewards = []
                images = []
                terminated_flags = []

                for step in range(max_steps):
                    # Get observation
                    raw_obs = self._get_obs()
                    obs = vec_normalize.normalize_obs(raw_obs) if vec_normalize else raw_obs

                    # Render image
                    img = self._render_image()

                    # Get action from policy
                    action, _ = policy.predict(obs, deterministic=True)

                    # Store pre-action data
                    observations.append(raw_obs)
                    actions.append(action)
                    images.append(img)

                    # Apply action
                    ctrl = action * self._mj_model.actuator_ctrlrange[:, 1]
                    self._mj_data.ctrl[:] = ctrl
                    for _ in range(frame_skip):
                        mujoco.mj_step(self._mj_model, self._mj_data)

                    # Compute reward (simplified -- just COM height + upright for filtering)
                    com_height = self._mj_data.subtree_com[0][2]
                    upright = self._mj_data.xmat[1].reshape(3, 3)[2, 2]
                    reward = com_height + upright
                    rewards.append(reward)

                    # Check termination
                    terminated = self._is_terminated()
                    terminated_flags.append(terminated)
                    if terminated:
                        break

                # Filter: only keep episodes that survived long enough
                if len(observations) < min_survival_steps:
                    continue

                # Save episode to HDF5
                ep_group = hf.create_group(f"episode_{kept:05d}")
                ep_group.attrs["language_command"] = lang_cmd
                ep_group.attrs["num_steps"] = len(observations)
                ep_group.attrs["survived"] = not terminated_flags[-1]

                ep_group.create_dataset(
                    "observations", data=np.array(observations, dtype=np.float32),
                    compression="gzip", compression_opts=4,
                )
                ep_group.create_dataset(
                    "actions", data=np.array(actions, dtype=np.float32),
                    compression="gzip", compression_opts=4,
                )
                ep_group.create_dataset(
                    "rewards", data=np.array(rewards, dtype=np.float32),
                    compression="gzip", compression_opts=4,
                )
                ep_group.create_dataset(
                    "images", data=np.array(images, dtype=np.uint8),
                    compression="gzip", compression_opts=4,
                )
                ep_group.create_dataset(
                    "terminated", data=np.array(terminated_flags, dtype=bool),
                )

                kept += 1

                if kept % 50 == 0:
                    print(f"[TrajectoryRecorder] {skill_id}: {kept} episodes kept "
                          f"({attempted} attempted)")

            # Update metadata
            meta.attrs["num_episodes"] = kept
            meta.attrs["total_attempted"] = attempted

        print(f"[TrajectoryRecorder] {skill_id}: Saved {kept}/{attempted} episodes "
              f"to {output_path}")
        return output_path

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
