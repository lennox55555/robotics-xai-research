"""
Simple Vision-Language-Action Model

A lightweight VLA that takes camera images, proprioceptive state, and
language commands to produce robot actions. Trained via behavioral cloning
from specialist RL policies (ATLAS Phase 4).

Architecture:
  - Vision: CNN encoder (128x128 RGB -> 256-dim)
  - Language: Learned skill embeddings (6 skills -> 128-dim)
  - Proprioception: MLP encoder (111-dim -> 128-dim)
  - Fusion: Concatenate [256 + 128 + 128 = 512] -> MLP -> 43-dim actions
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


SKILL_TO_ID = {
    "balance_stand": 0,
    "squat": 1,
    "jump": 2,
    "walk_forward": 3,
    "raise_right_hand": 4,
    "wave_right_hand": 5,
}

LANGUAGE_TO_SKILL = {
    "stand still and balance": 0, "maintain upright standing position": 0,
    "balance without moving": 0, "hold a stable standing pose": 0,
    "squat down": 1, "lower into a squat position": 1,
    "crouch down and hold": 1, "perform a deep squat": 1,
    "jump up": 2, "perform a vertical jump": 2,
    "jump and land": 2, "leap upward": 2,
    "walk forward": 3, "move forward at a moderate pace": 3,
    "take steps forward": 3, "walk straight ahead": 3,
    "raise your right hand": 4, "lift the right hand above your head": 4,
    "put your right hand up": 4, "raise right arm": 4,
    "wave your right hand": 5, "wave hello with your right hand": 5,
    "wave at me": 5, "wave goodbye": 5,
}


class SimpleVLA(nn.Module):
    """Vision-Language-Action model for multi-skill robot control."""

    def __init__(self, proprio_dim=111, action_dim=43, num_skills=6):
        super().__init__()

        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.num_skills = num_skills

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

    def predict_action(self, image, proprio, language_command: str):
        """Convenience method: takes a language string and returns action."""
        skill_id = LANGUAGE_TO_SKILL.get(language_command.lower(), 0)
        skill_tensor = torch.LongTensor([skill_id]).to(next(self.parameters()).device)

        if not isinstance(image, torch.Tensor):
            image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.FloatTensor(proprio).unsqueeze(0)

        with torch.no_grad():
            action = self(image, proprio, skill_tensor)
        return action.squeeze(0).numpy()


def load_vla(model_path: Optional[str] = None) -> SimpleVLA:
    """Load the trained VLA model."""
    if model_path is None:
        project_root = Path(__file__).parent.parent.parent
        model_path = str(project_root / "experiments" / "atlas" / "phase4_vla" / "final_model_full.pt")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = checkpoint["architecture"]

    model = SimpleVLA(
        proprio_dim=arch["proprio_dim"],
        action_dim=arch["action_dim"],
        num_skills=arch["num_skills"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
