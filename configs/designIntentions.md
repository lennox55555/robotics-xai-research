# Configs Directory Design Intentions

## Purpose

The `configs/` directory stores **experiment configurations** in YAML format. These define training parameters, environment settings, and logging options.

## Structure

```
configs/
├── default.yaml          # Default configuration (used if none specified)
├── experiments/          # Experiment-specific configs
│   ├── walking_v1.yaml
│   └── jumping_curriculum.yaml
└── designIntentions.md   # This file
```

## Configuration Schema

```yaml
# Experiment identification
experiment:
  name: "experiment_name"
  seed: 42

# Environment settings
env:
  name: "G1Humanoid-v0"       # Gymnasium environment ID
  max_episode_steps: 1000
  normalize_obs: true
  normalize_reward: true

# Training hyperparameters
training:
  algorithm: "PPO"            # PPO, SAC, TD3
  total_timesteps: 1000000
  learning_rate: 3e-4
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5

# Policy network architecture
model:
  policy: "MlpPolicy"
  hidden_sizes: [256, 256]
  activation: "tanh"

# Explainability settings
explainability:
  enabled: true
  method: "attention"         # attention, gradcam, shap
  log_frequency: 10000

# Logging configuration
logging:
  wandb:
    enabled: true
    project: "robotics-xai-research"
    tags: ["mujoco", "rl", "explainability"]
  save_frequency: 50000       # Save checkpoint every N steps
  eval_frequency: 10000       # Evaluate every N steps
  n_eval_episodes: 10
```

## Design Decisions

### Why YAML?

**Requirements**:
- Human-readable and editable
- Supports comments
- Hierarchical structure
- Standard format

**YAML**: Meets all requirements, cleaner than JSON for configs.

### Why OmegaConf?

We use OmegaConf for config management:
- Type checking
- Interpolation (`${experiment.name}`)
- Merging multiple configs
- Environment variable support

### Why Separate from Skill Configs?

**Experiment configs**: How to train (hyperparameters, logging)
**Skill configs**: What to train (reward function, success criteria)

They serve different purposes and change independently.

## Usage

```python
from omegaconf import OmegaConf

# Load default config
config = OmegaConf.load("configs/default.yaml")

# Override with experiment config
exp_config = OmegaConf.load("configs/experiments/walking_v1.yaml")
config = OmegaConf.merge(config, exp_config)

# Override via command line
# python train.py training.learning_rate=1e-4
```

## Environment Variables

Sensitive values via environment:
```yaml
logging:
  wandb:
    api_key: ${oc.env:WANDB_API_KEY}
```

## Config Inheritance

Experiments inherit from default:
```yaml
# configs/experiments/walking_v1.yaml
defaults:
  - ../default

# Only override what changes
training:
  total_timesteps: 2000000
  learning_rate: 1e-4

experiment:
  name: "walking_v1"
```

## Best Practices

1. **Never commit secrets** - Use environment variables
2. **Document experiments** - Add comments explaining choices
3. **Version configs** - Track config changes in git
4. **Use meaningful names** - `walking_curriculum_v2.yaml` not `config2.yaml`
