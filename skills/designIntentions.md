# Skills Directory Design Intentions

## Purpose

The `skills/` directory stores all **skill definitions and trained models**. It serves as the skill library that agents query and update.

## Structure

```
skills/
├── configs/              # Skill definitions (JSON)
│   ├── balance.json      # Skill metadata and config
│   ├── walk_forward.json
│   └── jump.json
│
├── trained/              # Trained models
│   ├── balance/
│   │   ├── model.zip           # Policy weights (SB3 format)
│   │   ├── env_normalize.pkl   # Observation normalizer
│   │   └── checkpoints/        # Training checkpoints
│   │       ├── model_50000_steps.zip
│   │       └── model_100000_steps.zip
│   │
│   └── walk_forward/
│       ├── model.zip
│       └── checkpoints/
│
└── logs/                 # Training logs
    ├── balance/
    │   └── evaluations.npz
    └── tensorboard/      # TensorBoard logs
```

## Skill Definition Format

```json
{
  "skill_id": "walk_forward",
  "name": "Walk Forward",
  "description": "Walk forward at moderate speed while maintaining balance",

  "reward_components": [
    "velocity_forward",
    "upright_reward",
    "energy_efficiency"
  ],

  "success_criteria": "forward_velocity > 0.5 m/s for 100 steps",

  "termination_conditions": [
    "com_height < 0.4",
    "episode_length > 1000"
  ],

  "prerequisites": ["balance"],

  "config": {
    "algorithm": "PPO",
    "total_timesteps": 500000,
    "learning_rate": 3e-4,
    "transfer_from": "balance"
  },

  "status": "trained",

  "training_metrics": {
    "mean_reward": 245.3,
    "total_episodes": 1523,
    "training_time_hours": 2.5
  }
}
```

## Design Decisions

### Why JSON for Configs?

**Requirements**:
- Human-readable
- Easy to edit manually
- Parseable by agents

**JSON**: Meets all requirements, ubiquitous support.

### Why Separate configs/ and trained/?

**Problem**: Skill definitions should exist before training completes.

**Solution**:
- `configs/`: Created when skill is defined (status: "pending")
- `trained/`: Created when training completes
- Status field tracks current state

### Why checkpoints/?

**Problem**: Training can fail or be interrupted.

**Solution**: Save checkpoints every N steps:
- Resume from checkpoint
- Analyze learning progress
- Compare different training stages

### Why env_normalize.pkl?

**Problem**: Observation normalization must match between training and execution.

**Solution**: Save VecNormalize state alongside model.

## Skill Lifecycle

```
1. DEFINED
   Learning Agent creates config JSON
   Status: "pending"

2. TRAINING
   SkillTrainer starts training
   Status: "training"
   Checkpoints saved periodically

3. TRAINED
   Training complete
   Status: "trained"
   model.zip saved
   Metrics recorded

4. EVALUATED
   Performance Agent tests skill
   Evaluation metrics added

5. ANALYZED
   Research Agent explains policy
   Insights recorded
```

## Querying Skills

```python
from src.skill_learning.skill import SkillLibrary

library = SkillLibrary(Path("skills"))

# Get all trained skills
trained = library.get_trained_skills()

# Find skills for transfer learning
similar = library.find_similar_skills(new_skill)

# Load specific skill
skill = library.get_skill("walk_forward")
```

## File Naming Conventions

- Skill IDs: `snake_case` (e.g., `walk_forward`)
- Checkpoint names: `model_{steps}_steps.zip`
- No spaces or special characters

## Persistence

- All changes persisted immediately
- JSON configs are human-editable
- Models are SB3-compatible zip files
