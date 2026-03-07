# Explainability (XAI) Design Intentions

## Purpose

The `explainability/` module provides **interpretability tools** for understanding learned robot policies. This is crucial for:
- Debugging why skills fail
- Building trust in learned behaviors
- Improving reward function design
- Research into robot learning

## Core Question

> "Why did the robot take that action in that state?"

## XAI Methods Implemented

### 1. Saliency Maps (Input Gradients)

**What**: Compute gradient of action with respect to input observations.

**Shows**: Which observation dimensions most influence the policy.

**Use case**: "Which joints is the robot paying attention to?"

```python
saliency = analyzer.compute_saliency(observation)
# saliency[i] = importance of observation dimension i
```

### 2. Feature Importance (Aggregated)

**What**: Average saliency across many observations.

**Shows**: Overall feature importance for the policy.

**Use case**: "Does the policy use contact sensors?"

```python
importance = analyzer.compute_feature_importance(observations)
# importance["mean_importance"] = average importance per feature
```

### 3. Action Distribution Analysis

**What**: Sample actions from policy, compute statistics.

**Shows**: How deterministic/uncertain the policy is.

**Use case**: "Is the policy confident in this state?"

```python
analysis = analyzer.analyze_action_distribution(observation)
# analysis["entropy"] = policy uncertainty
# analysis["action_std"] = action variance
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PolicyAnalyzer                            │
├─────────────────────────────────────────────────────────────┤
│  Input: Trained policy (Stable-Baselines3 model)            │
│  Output: Explanations in various formats                    │
├─────────────────────────────────────────────────────────────┤
│  Methods:                                                   │
│  - compute_saliency(obs) → per-feature importance           │
│  - compute_feature_importance(obs_batch) → aggregated       │
│  - analyze_action_distribution(obs) → policy statistics     │
│  - analyze_and_log() → full report to W&B                   │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why Gradient-Based Methods?

**Alternatives**:
- SHAP: More theoretically grounded but slow
- LIME: Local approximations, can be unstable
- Attention: Only for attention-based policies

**Choice**: Gradients are fast and work with any differentiable policy.

### Why Focus on Observation Importance?

**Problem**: Robot policies have many inputs (44+ joint positions/velocities).

**Insight**: Understanding which inputs matter reveals policy strategy:
- Policy ignores finger joints → not using hands
- Policy focuses on ankle angles → balance-focused
- Policy uses velocity more than position → reactive control

### Why Aggregate Across Episodes?

**Problem**: Single-observation analysis is noisy.

**Solution**: Collect importance across diverse states, compute statistics:
```python
# Run 100 episodes, collect observations
all_obs = collect_episode_observations(policy, env, n_episodes=100)

# Aggregate importance
importance = analyzer.compute_feature_importance(all_obs)
```

## Integration with Research Agent

The Research Agent uses these tools via prompts like:

```
"Why does the walking skill fall when turning?"

→ Research Agent:
  1. Collects observations from failure cases
  2. Runs saliency analysis
  3. Compares to successful cases
  4. Reports: "Policy over-relies on forward velocity,
     ignores lateral balance indicators"
```

## Output Formats

### For LLM Agents (JSON)
```json
{
  "top_features": [
    {"index": 12, "name": "left_ankle_pitch", "importance": 0.45},
    {"index": 0, "name": "root_velocity_x", "importance": 0.32}
  ],
  "policy_entropy": 1.23,
  "action_variance": [0.1, 0.2, ...]
}
```

### For Humans (Visualizations)
- Bar charts of feature importance
- Heatmaps over time
- Action distribution plots

### For W&B (Logged Metrics)
- `feature_importance/mean`
- `policy/entropy`
- `model/total_params`

## Future Extensions

- **Counterfactual explanations**: "What would change the action?"
- **Concept-based explanations**: "Is the robot 'balancing'?"
- **Temporal explanations**: "Why did it start falling?"
- **Comparative explanations**: "Why is skill A better than B?"

## Research Background

- **Saliency**: Simonyan et al., "Deep Inside Convolutional Networks" (2014)
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
- **Policy Explainability**: Greydanus et al., "Visualizing and Understanding Atari Agents" (2018)
