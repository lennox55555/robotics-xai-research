# MuJoCo Experiments

This directory contains experiments using MuJoCo environments for robotics RL research.

## Available Environments

Common MuJoCo environments for robotics research:
- `Ant-v4` - Quadruped locomotion
- `HalfCheetah-v4` - Planar biped running
- `Hopper-v4` - Single-legged hopping
- `Walker2d-v4` - Bipedal walking
- `Humanoid-v4` - Full humanoid control
- `Swimmer-v4` - Swimming locomotion
- `InvertedPendulum-v4` - Balancing task
- `InvertedDoublePendulum-v4` - Double pendulum balancing
- `Reacher-v4` - 2D reaching task
- `Pusher-v4` - Object pushing

## Running Experiments

### Basic Training

```bash
# From project root
python experiments/mujoco/train.py

# With custom config
python experiments/mujoco/train.py --config configs/my_experiment.yaml
```

### Monitor on W&B

Training metrics are automatically logged to Weights & Biases. View your experiments at:
https://wandb.ai/

## Creating New Experiments

1. Create a new config in `configs/`
2. Modify environment, algorithm, and hyperparameters
3. Run training with your config

## Explainability Analysis

The training script automatically runs explainability analysis including:
- Feature importance via saliency maps
- Action distribution analysis
- Policy behavior visualization
