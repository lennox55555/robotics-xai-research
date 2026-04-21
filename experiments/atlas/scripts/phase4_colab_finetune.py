"""
ATLAS Phase 4: VLA Fine-Tuning on Google Colab (A100)

Run this script in Google Colab after:
1. Cloning the repo: git clone -b colab-phase4-vla https://github.com/lennox55555/robotics-xai-research.git
2. Uploading atlas_trajectories.zip to Google Drive
3. Mounting Google Drive in Colab

Usage in Colab cell:
    !python robotics-xai-research/experiments/atlas/scripts/phase4_colab_finetune.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path


def setup_environment():
    """Install dependencies and set up paths."""
    print("=" * 60)
    print("ATLAS Phase 4: VLA Fine-Tuning Setup")
    print("=" * 60)

    # Install dependencies in stages to handle failures gracefully
    print("\nInstalling core dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "h5py", "wandb", "torch", "torchvision",
    ], check=True)

    # Try installing Octo and its dependencies from GitHub
    print("Installing Octo from GitHub...")
    octo_result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "dlimp @ git+https://github.com/kvablack/dlimp.git",
        "git+https://github.com/octo-models/octo.git",
    ], capture_output=True, text=True)

    if octo_result.returncode != 0:
        print(f"Octo install failed (will use PyTorch fallback): {octo_result.stderr[:200]}")
    else:
        print("Octo installed successfully")

    print("Dependencies installed.")


def mount_and_unzip():
    """Mount Google Drive and unzip trajectory data."""
    from google.colab import drive

    print("\nMounting Google Drive...")
    drive.mount("/content/drive")

    # Find the zip file
    zip_candidates = [
        "/content/drive/MyDrive/atlas_trajectories.zip",
        "/content/drive/My Drive/atlas_trajectories.zip",
    ]

    zip_path = None
    for candidate in zip_candidates:
        if os.path.exists(candidate):
            zip_path = candidate
            break

    if zip_path is None:
        print("ERROR: atlas_trajectories.zip not found in Google Drive root.")
        print("Please upload it to the root of your Google Drive.")
        print("Looked in:", zip_candidates)
        return False

    # Unzip to project directory
    project_root = "/content/robotics-xai-research"
    raw_dir = f"{project_root}/experiments/atlas/phase3_trajectories/raw"
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\nUnzipping {zip_path} to {raw_dir}...")
    print("This may take 10-15 minutes for 38 GB...")
    subprocess.run(["unzip", "-o", "-q", zip_path, "-d", raw_dir], check=True)

    # Verify
    hdf5_files = list(Path(raw_dir).glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files:")
    for f in sorted(hdf5_files):
        size_gb = f.stat().st_size / (1024**3)
        print(f"  {f.name}: {size_gb:.1f} GB")

    return True


def convert_hdf5_to_octo_dataset(project_root: str):
    """Convert HDF5 trajectory files to the format Octo expects.

    Octo fine-tuning expects a dataset as a list of episodes, each being
    a dict with 'observations', 'actions', and 'language_instruction'.
    """
    import h5py
    import numpy as np
    import pickle

    raw_dir = Path(project_root) / "experiments" / "atlas" / "phase3_trajectories" / "raw"
    output_dir = Path(project_root) / "experiments" / "atlas" / "phase3_trajectories" / "octo_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    skills = [
        "balance_stand", "squat", "jump",
        "walk_forward", "raise_right_hand", "wave_right_hand"
    ]

    all_episodes = []
    total_steps = 0

    for skill_id in skills:
        hdf5_path = raw_dir / f"{skill_id}_trajectories.hdf5"
        if not hdf5_path.exists():
            print(f"  SKIP: {skill_id} (no HDF5 file)")
            continue

        print(f"\nProcessing {skill_id}...")
        try:
            with h5py.File(str(hdf5_path), "r") as hf:
                num_episodes = hf["metadata"].attrs.get("num_episodes", 0)
                # Subsample: take up to 500 episodes per skill to keep manageable
                max_episodes = min(num_episodes, 500)
                loaded = 0

                for i in range(max_episodes):
                    ep_key = f"episode_{i:05d}"
                    if ep_key not in hf:
                        continue

                    try:
                        ep = hf[ep_key]
                        episode_data = {
                            "observations": {
                                "image_primary": np.array(ep["images"]),  # (T, 128, 128, 3)
                                "proprio": np.array(ep["observations"]),  # (T, 111)
                            },
                            "actions": np.array(ep["actions"]),  # (T, 43)
                            "language_instruction": ep.attrs["language_command"],
                        }
                        all_episodes.append(episode_data)
                        total_steps += len(ep["actions"])
                        loaded += 1
                    except Exception as ep_err:
                        print(f"  Warning: skipping {ep_key} ({ep_err})")
                        break  # File likely truncated, stop reading this skill

                print(f"  Loaded {loaded} episodes from {skill_id}")
        except Exception as e:
            print(f"  SKIP {skill_id}: corrupted file ({e})")

    print(f"\nTotal: {len(all_episodes)} episodes, {total_steps:,} steps")

    # Save as pickle for easy loading
    dataset_path = output_dir / "atlas_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(all_episodes, f)
    print(f"Dataset saved to: {dataset_path}")

    return all_episodes, dataset_path


def finetune_octo(project_root: str, dataset, num_steps: int = 50000):
    """Fine-tune Octo-base on the ATLAS trajectory dataset.

    Octo uses a transformer encoder + diffusion action head.
    We freeze the vision encoder and language encoder, and train
    the cross-attention layers + action head.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from functools import partial

    print("\n" + "=" * 60)
    print("Fine-Tuning Octo on ATLAS Dataset")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Dataset: {len(dataset)} episodes")
    print(f"Training steps: {num_steps:,}")

    try:
        from octo.model.octo_model import OctoModel

        # Load pretrained Octo-base
        print("\nLoading Octo-base model...")
        model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
        print("Octo-base loaded successfully")

        # Configure fine-tuning
        # Octo's fine-tuning API handles:
        # - Freezing vision/language encoders
        # - Training cross-attention + action head
        # - Adapting to our 43-DOF action space (vs Octo's default 7-DOF)

        from octo.utils.train_callbacks import SaveCallback
        from octo.utils.train_utils import process_batch

        output_dir = Path(project_root) / "experiments" / "atlas" / "phase4_vla" / "octo"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create data iterator from our episodes
        def data_iterator(episodes, batch_size=32):
            """Yields batches of training data in Octo's expected format."""
            indices = np.arange(len(episodes))
            while True:
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start:start + batch_size]
                    if len(batch_indices) < batch_size:
                        continue

                    batch_images = []
                    batch_proprio = []
                    batch_actions = []
                    batch_language = []

                    for idx in batch_indices:
                        ep = episodes[idx]
                        T = len(ep["actions"])
                        # Random window of 2 steps (Octo uses short action chunks)
                        t = np.random.randint(0, max(1, T - 2))
                        window = min(2, T - t)

                        batch_images.append(ep["observations"]["image_primary"][t])
                        batch_proprio.append(ep["observations"]["proprio"][t])
                        batch_actions.append(ep["actions"][t:t+window].mean(axis=0))
                        batch_language.append(ep["language_instruction"])

                    yield {
                        "image_primary": np.stack(batch_images),
                        "proprio": np.stack(batch_proprio),
                        "actions": np.stack(batch_actions),
                        "language": batch_language,
                    }

        # Fine-tune using Octo's built-in training
        print("\nStarting fine-tuning...")
        print(f"Output: {output_dir}")

        # Octo's finetune_new_obs_action method handles the architecture adaptation
        finetuned_model = model.finetune(
            dataset_iterator=data_iterator(dataset),
            num_steps=num_steps,
            action_dim=43,  # Our G1 action space
            action_type="continuous",
            learning_rate=3e-4,
            save_dir=str(output_dir / "checkpoints"),
            save_interval=5000,
        )

        # Save final model
        final_path = output_dir / "final_model"
        finetuned_model.save_pretrained(str(final_path))
        print(f"\nFinal model saved to: {final_path}")

        return finetuned_model

    except ImportError as e:
        print(f"\nOcto import failed: {e}")
        print("Falling back to custom transformer training...")
        return finetune_custom_vla(project_root, dataset, num_steps)

    except Exception as e:
        print(f"\nOcto fine-tuning failed: {e}")
        print("This may be due to API changes in Octo.")
        print("Falling back to custom transformer training...")
        return finetune_custom_vla(project_root, dataset, num_steps)


def finetune_custom_vla(project_root: str, dataset, num_steps: int = 50000):
    """Fallback: Train a custom vision-language-action model using PyTorch.

    Architecture:
    - Vision: ResNet-18 (pretrained) encodes 128x128 image -> 512-dim
    - Language: Sentence embedding (384-dim)
    - Fusion: Concatenate [vision, language, proprio] -> MLP -> 43-dim actions

    Simpler than Octo but proves the distillation concept.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    print("\n" + "=" * 60)
    print("Training Custom VLA (PyTorch Fallback)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Simple VLA architecture
    class SimpleVLA(nn.Module):
        def __init__(self, proprio_dim=111, action_dim=43, num_skills=6):
            super().__init__()

            # Vision encoder (lightweight CNN)
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(),
            )

            # Language encoder (learned embeddings per skill)
            self.language_encoder = nn.Embedding(num_skills, 128)

            # Proprio encoder
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            # Fusion + action head
            # 256 (vision) + 128 (language) + 128 (proprio) = 512
            self.action_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh(),  # Actions in [-1, 1]
            )

        def forward(self, image, proprio, language_id):
            vis_feat = self.vision_encoder(image)
            lang_feat = self.language_encoder(language_id)
            prop_feat = self.proprio_encoder(proprio)
            fused = torch.cat([vis_feat, lang_feat, prop_feat], dim=-1)
            return self.action_head(fused)

    # Dataset
    SKILL_TO_ID = {
        "balance_stand": 0, "squat": 1, "jump": 2,
        "walk_forward": 3, "raise_right_hand": 4, "wave_right_hand": 5,
    }

    # Map language commands to skill IDs
    COMMAND_TO_SKILL = {}
    _LANGUAGE_COMMANDS = {
        "balance_stand": ["stand still and balance", "maintain upright standing position", "balance without moving", "hold a stable standing pose"],
        "squat": ["squat down", "lower into a squat position", "crouch down and hold", "perform a deep squat"],
        "jump": ["jump up", "perform a vertical jump", "jump and land", "leap upward"],
        "walk_forward": ["walk forward", "move forward at a moderate pace", "take steps forward", "walk straight ahead"],
        "raise_right_hand": ["raise your right hand", "lift the right hand above your head", "put your right hand up", "raise right arm"],
        "wave_right_hand": ["wave your right hand", "wave hello with your right hand", "wave at me", "wave goodbye"],
    }
    for skill_id, commands in _LANGUAGE_COMMANDS.items():
        for cmd in commands:
            COMMAND_TO_SKILL[cmd] = SKILL_TO_ID.get(skill_id, 0)

    class TrajectoryDataset(Dataset):
        def __init__(self, episodes):
            self.samples = []
            for ep in episodes:
                T = len(ep["actions"])
                lang = ep["language_instruction"]
                skill_id = COMMAND_TO_SKILL.get(lang, 0)
                for t in range(T):
                    self.samples.append({
                        "image": ep["observations"]["image_primary"][t],
                        "proprio": ep["observations"]["proprio"][t],
                        "action": ep["actions"][t],
                        "skill_id": skill_id,
                    })
            print(f"Dataset: {len(self.samples):,} samples from {len(episodes)} episodes")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            image = torch.FloatTensor(s["image"]).permute(2, 0, 1) / 255.0  # (3, 128, 128)
            proprio = torch.FloatTensor(s["proprio"])
            action = torch.FloatTensor(s["action"])
            skill_id = torch.LongTensor([s["skill_id"]])[0]
            return image, proprio, skill_id, action

    # Build dataset and dataloader
    print("\nBuilding dataset...")
    train_dataset = TrajectoryDataset(dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    # Build model
    model = SimpleVLA(proprio_dim=111, action_dim=43, num_skills=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    criterion = nn.MSELoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training loop
    print(f"\nTraining for {num_steps:,} steps...")
    output_dir = Path(project_root) / "experiments" / "atlas" / "phase4_vla" / "custom_vla"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()

    while step < num_steps:
        epoch += 1
        for batch in train_loader:
            if step >= num_steps:
                break

            images, proprios, skill_ids, actions = [x.to(device) for x in batch]

            pred_actions = model(images, proprios, skill_ids)
            loss = criterion(pred_actions, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if step % 1000 == 0:
                avg_loss = running_loss / 1000
                elapsed = time.time() - start_time
                fps = step / elapsed
                print(f"Step {step:>6,}/{num_steps:,} | loss={avg_loss:.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | {fps:.0f} steps/s")
                running_loss = 0.0

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), output_dir / "best_model.pt")

            if step % 10000 == 0:
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, output_dir / f"checkpoint_{step:06d}.pt")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "architecture": {
            "proprio_dim": 111,
            "action_dim": 43,
            "num_skills": 6,
            "skill_names": list(SKILL_TO_ID.keys()),
        },
    }, output_dir / "final_model_full.pt")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")

    return model


def main():
    project_root = "/content/robotics-xai-research"

    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
        project_root = str(Path(__file__).parent.parent.parent.parent)

    # Step 1: Setup
    setup_environment()

    # Step 2: Verify data exists (mount Drive and unzip manually in Colab cells first)
    raw_dir = Path(project_root) / "experiments" / "atlas" / "phase3_trajectories" / "raw"
    hdf5_files = list(raw_dir.glob("*.hdf5")) if raw_dir.exists() else []
    if not hdf5_files:
        print("\nERROR: No HDF5 files found in:")
        print(f"  {raw_dir}")
        print("\nIn Colab, run these cells BEFORE this script:")
        print("  Cell 1: from google.colab import drive; drive.mount('/content/drive')")
        print("  Cell 2: !unzip -o -q /content/drive/MyDrive/atlas_trajectories.zip -d /content/robotics-xai-research/experiments/atlas/phase3_trajectories/raw/")
        return
    else:
        print(f"\nFound {len(hdf5_files)} HDF5 files:")
        for f in sorted(hdf5_files):
            size_gb = f.stat().st_size / (1024**3)
            print(f"  {f.name}: {size_gb:.1f} GB")

    # Step 3: Convert data
    print("\n" + "=" * 60)
    print("Converting HDF5 to training dataset...")
    print("=" * 60)
    sys.path.insert(0, project_root)
    dataset, dataset_path = convert_hdf5_to_octo_dataset(project_root)

    # Step 4: Fine-tune
    model = finetune_octo(project_root, dataset, num_steps=50000)

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print("Next: Download the trained model and run Phase 5 (XAI) locally")


if __name__ == "__main__":
    main()
