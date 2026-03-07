# Skill Learning Design Intentions

## Purpose

The `skill_learning/` module provides the **reinforcement learning infrastructure** for training robot skills. It bridges LLM-generated skill definitions with actual RL training.

## Core Concept: What is a Skill?

A **Skill** is a learnable behavior that:
- Has a clear objective (e.g., "walk forward")
- Is defined by a reward function
- Can be trained in isolation
- Can be composed with other skills
- Can transfer knowledge to related skills

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SKILL DEFINITION                          │
│  (Created by Learning Agent)                                 │
│                                                             │
│  skill_id: walk_forward                                     │
│  reward_components: [velocity_forward, upright, efficiency] │
│  success_criteria: velocity > 0.5 m/s                       │
│  prerequisites: [balance]                                   │
│  transfer_from: balance                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SKILL TRAINER                             │
│  (Converts definition to RL experiment)                      │
│                                                             │
│  1. Create environment with skill-specific reward           │
│  2. Initialize policy (or load from transfer source)        │
│  3. Run PPO/SAC/TD3 training                                │
│  4. Save checkpoints and final model                        │
│  5. Update skill status and metrics                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRAINED SKILL                             │
│  (Saved in skills/trained/)                                  │
│                                                             │
│  skills/trained/walk_forward/                               │
│  ├── model.zip           # Policy weights                   │
│  ├── env_normalize.pkl   # Observation normalizer           │
│  └── checkpoints/        # Training checkpoints             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### `skill.py` - Skill Definitions

```python
@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    reward_components: List[str]
    success_criteria: str
    prerequisites: List[str]
    config: SkillConfig  # Training hyperparameters
    status: SkillStatus  # pending, training, trained, failed
```

### `skill_trainer.py` - RL Training Engine

Wraps Stable-Baselines3 with:
- Dynamic reward function construction
- Transfer learning support
- Progress tracking and callbacks
- Evaluation routines

## Design Decisions

### Why Skill-Based Rather Than End-to-End?

**Problem**: Training complex behaviors (walk + jump + backflip) end-to-end is extremely difficult.

**Solution**: Hierarchical skill learning:
1. Train foundational skills (balance)
2. Build on them (walk uses balance)
3. Compose for complex behaviors

**Benefits**:
- Faster training per skill
- Reusable skill library
- Transfer learning opportunities
- Easier debugging

### Why Reward Components?

**Problem**: Designing reward functions is hard. Agents often find "reward hacks."

**Solution**: Composable reward components:
```python
REWARD_COMPONENTS = {
    "height_reward": lambda env, data: data.com_height,
    "upright_reward": lambda env, data: data.torso_orientation,
    "velocity_forward": lambda env, data: data.root_velocity[0],
    "energy_efficiency": lambda env, data: -sum(data.actuator_forces**2),
}
```

The Learning Agent selects and weights these components.

### Why Transfer Learning?

**Problem**: Each skill trained from scratch is slow.

**Solution**: Initialize new skills from related trained skills:
- `jump` initialized from `balance` (shares stability)
- `run` initialized from `walk` (similar gait)

### Why Skill Library?

**Problem**: Need to track what's trained, reuse skills, find transfer candidates.

**Solution**: `SkillLibrary` class:
- Persists skill definitions to disk
- Finds similar skills for transfer
- Tracks training status

## Training Configuration

```yaml
# Default training config
algorithm: PPO
total_timesteps: 500000
learning_rate: 3e-4
batch_size: 64
gamma: 0.99
gae_lambda: 0.95

# For complex skills
curriculum:
  - stage: 1
    max_velocity: 0.2
    timesteps: 200000
  - stage: 2
    max_velocity: 0.5
    timesteps: 300000
```

## File Structure

```
skills/
├── configs/              # Skill definitions (JSON)
│   ├── balance.json
│   ├── walk_forward.json
│   └── jump.json
├── trained/              # Trained models
│   ├── balance/
│   │   ├── model.zip
│   │   └── checkpoints/
│   └── walk_forward/
│       ├── model.zip
│       └── checkpoints/
└── logs/                 # Training logs
    └── tensorboard/
```

## Integration Points

- **Learning Agent**: Creates skill definitions
- **Performance Agent**: Loads and executes trained skills
- **Research Agent**: Analyzes training curves and policies
- **MCP Servers**: Expose training/evaluation as tools
