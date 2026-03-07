# Robotics XAI Research

**LLM-Orchestrated Multi-Agent System for Teaching Humanoid Robots New Skills**

A sophisticated research platform combining large language models with reinforcement learning to teach a Unitree G1 humanoid robot complex motor skills, with built-in explainability (XAI).

## Architecture

```
+-------------------------------------------------------------+
|                         USER INPUT                           |
|              "Teach the robot to walk and jump"              |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
|                       ORCHESTRATOR                           |
|              Coordinates specialized agents                   |
+-------------------------------------------------------------+
                |               |               |
    +-----------+               |               +-----------+
    v                           v                           v
+-------------+         +-------------+         +-------------+
|   LEARNING  |         |  PERFORM.   |         |   RESEARCH  |
|    AGENT    |         |    AGENT    |         |    AGENT    |
|             |         |             |         |             |
| Trains RL   |         | Runs sims   |         | XAI &       |
| policies    |         | Tests skills|         | Analysis    |
+-------------+         +-------------+         +-------------+
```

## Key Features

- **Multi-Agent System**: Specialized agents for training, testing, and analysis
- **Hybrid Context Engineering**: RAG + sliding window for optimal LLM context
- **Unitree G1 Humanoid**: Full 44-DOF robot with dexterous hands
- **Explainable AI**: Understand why policies make decisions
- **MuJoCo Simulation**: High-fidelity physics simulation
- **Full Logging**: Every conversation and training run recorded

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/robotics-xai-research.git
cd robotics-xai-research

# Run setup
./setup.sh
source .venv/bin/activate

# Add your Anthropic API key to .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env

# Start the orchestrator
python run_orchestrator.py
```

## Example Interaction

```
You: Teach the robot to walk forward

Orchestrator: I'll break this down into learnable skills:
1. balance - Maintain upright posture
2. walk_forward - Locomotion with stable gait

HANDOFF TO: learning_agent

TASK: Create skill definitions for balance and walking...

Learning Agent: I've designed the following skills:
- balance: reward = height + upright - energy
- walk_forward: reward = velocity + upright + efficiency
  (transfers from balance)

Starting training for 'balance'...
```

## Project Structure

```
robotics-xai-research/
├── run_orchestrator.py      # Main entry point
├── src/
│   ├── agents/              # Multi-agent system
│   ├── context/             # RAG + context engineering
│   ├── mcp_servers/         # MCP tool servers
│   ├── skill_learning/      # RL training infrastructure
│   ├── envs/                # Gymnasium environments
│   └── explainability/      # XAI tools
├── skills/                  # Trained models & definitions
├── logs/                    # Conversation & training logs
├── configs/                 # Experiment configurations
└── mujoco_menagerie/        # Robot models
```

## Documentation

Each directory contains a `designIntentions.md` file explaining its purpose and architecture. Start with:

- [`DOCUMENTATION.md`](DOCUMENTATION.md) - Documentation guide
- [`src/designIntentions.md`](src/designIntentions.md) - Source code overview

## Configuration

Edit `configs/default.yaml` to customize:
- Training hyperparameters
- Environment settings
- Logging options
- XAI methods

## Conversation Logs

All interactions are saved to `logs/conversations/`:
- **JSON**: Machine-readable format
- **Markdown**: Human-readable format

```bash
# View conversation history
python run_orchestrator.py --history

# View specific session
python run_orchestrator.py --view conv_20240315_143022_abc123
```

## Contributing

This is a research project. Contributions welcome!

1. Read the `designIntentions.md` files to understand architecture
2. Follow existing patterns and conventions
3. Add documentation for new features

## License

MIT License - See LICENSE file

## Acknowledgments

- [Anthropic](https://anthropic.com) - Claude LLM
- [MuJoCo](https://mujoco.org) - Physics simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io) - RL algorithms
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Robot models
