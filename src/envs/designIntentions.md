# Environments Design Intentions

## Purpose

The `envs/` module provides **Gymnasium-compatible environments** for robot simulation. The primary environment wraps the Unitree G1 humanoid in MuJoCo.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    G1HumanoidEnv                             │
│               (Gymnasium Environment)                        │
├─────────────────────────────────────────────────────────────┤
│  Observation Space: Box(111,)                                │
│  - Joint positions (44)                                      │
│  - Joint velocities (44)                                     │
│  - COM position (3)                                          │
│  - Torso orientation (9)                                     │
│  - Contact sensors (11)                                      │
├─────────────────────────────────────────────────────────────┤
│  Action Space: Box(43,)                                      │
│  - Torque commands for all actuators                         │
│  - Normalized to [-1, 1]                                     │
├─────────────────────────────────────────────────────────────┤
│  Reward: Configurable via skill definition                   │
│  - Default: height + upright - energy                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MuJoCo Physics                            │
│                                                             │
│  Model: mujoco_menagerie/unitree_g1/g1_with_hands.xml       │
│  Timestep: 0.002s (500 Hz physics)                          │
│  Frame skip: 5 (100 Hz control)                             │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why Custom Environment (Not gym.make)?

**Problem**: Standard Gymnasium humanoid environments (Humanoid-v4) use a generic humanoid model, not the Unitree G1.

**Solution**: Custom environment that:
- Loads G1 model from MuJoCo Menagerie
- Provides G1-specific observations (finger joints)
- Supports skill-specific reward functions

### Why Frame Skip = 5?

**Physics**: MuJoCo runs at 500 Hz for stability.
**Control**: Robot controlled at 100 Hz (realistic).
**Frame skip**: 5 physics steps per control step.

### Why Normalized Actions?

**Problem**: Different actuators have different torque limits.

**Solution**: Actions in [-1, 1], scaled to actuator limits:
```python
scaled_action = ctrl_range[:, 0] + (action + 1) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
```

### Why Configurable Rewards?

**Problem**: Different skills need different rewards.

**Solution**: Environment accepts reward components from skill definition:
```python
env = G1HumanoidEnv(
    reward_components=["velocity_forward", "upright_reward"],
    reward_weights=[1.0, 0.5],
)
```

## Robot Specification

| Property | Value |
|----------|-------|
| Model | Unitree G1 with dexterous hands |
| Total joints | 44 |
| Actuators | 43 |
| Height | ~1.27m |
| Mass | ~35kg |

### Joint Groups
- **Legs**: 12 DOF (hip pitch/roll/yaw, knee, ankle pitch/roll × 2)
- **Torso**: 3 DOF (waist yaw/roll/pitch)
- **Arms**: 14 DOF (shoulder, elbow, wrist × 2)
- **Hands**: 14 DOF (thumb, index, middle × 2)

## Observation Details

```python
observation = np.concatenate([
    qpos,           # Joint positions (44)
    qvel,           # Joint velocities (44)
    com_pos,        # Center of mass position (3)
    torso_xmat,     # Torso rotation matrix (9)
    # Optional: contact forces, IMU, etc.
])
```

## Render Modes

- `"human"`: Opens MuJoCo viewer window
- `"rgb_array"`: Returns pixel array (for video recording)
- `None`: No rendering (fastest for training)

## Usage

```python
from src.envs.g1_humanoid import G1HumanoidEnv

# Basic usage
env = G1HumanoidEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# With rendering
env = G1HumanoidEnv(render_mode="human")

# With custom reward (for skill training)
env = G1HumanoidEnv(
    task="walk",
    reward_components=["velocity_forward", "energy_efficiency"],
)
```

## Future Extensions

- **Domain randomization**: Vary mass, friction, etc.
- **Terrain**: Add slopes, stairs, obstacles
- **Objects**: Add manipulation objects
- **Multi-robot**: Multiple G1s in same scene
