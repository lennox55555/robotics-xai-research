# Source Code Design Intentions

## Purpose

The `src/` directory contains all source code for the multi-agent robot skill learning system. This is a research platform for exploring LLM-orchestrated reinforcement learning with explainable AI.

## Architecture Overview

```
src/
├── agents/          # Multi-agent system (LLM-based)
├── context/         # Context engineering (RAG + sliding window)
├── mcp_servers/     # MCP tool servers for each agent
├── skill_learning/  # RL training infrastructure
├── envs/            # Gymnasium environments (MuJoCo)
├── explainability/  # XAI / interpretability tools
├── transfer/        # Transfer learning utilities
├── orchestrator/    # Task decomposition (legacy, see agents/)
└── utils/           # Logging, callbacks, helpers
```

## Design Principles

### 1. **Each Agent is Autonomous**
Each agent (Learning, Performance, Research) has:
- Its own Claude LLM instance
- Its own context window
- Its own system prompt optimized for its role
- Access to shared long-term memory via RAG

**Why**: Specialized prompts dramatically improve agent performance. A learning agent needs different context than a research agent.

### 2. **Hybrid Context Engineering**
We combine two approaches:
- **RAG (Retrieval-Augmented Generation)**: Vector store for long-term memory
- **Sliding Window**: Recent N messages for immediate context

**Why**: Pure RAG misses recent conversation flow. Pure sliding window forgets important history. The hybrid approach gives both.

### 3. **Structured Handoffs**
When the orchestrator delegates to an agent, it sends:
- Specific task instructions
- Summarized relevant context
- Expected output format

**Why**: Raw conversation dumps overwhelm agents. Structured handoffs focus attention.

### 4. **Everything is Logged**
All conversations, decisions, and training runs are saved:
- JSON for machine readability
- Markdown for human readability

**Why**: Research requires reproducibility. Logs enable debugging and analysis.

### 5. **MCP for Tool Access**
Each agent's capabilities are exposed via MCP (Model Context Protocol) servers.

**Why**: Standardized tool interface. Can be used by external systems.

## Key Dependencies

| Module | Depends On | Provides To |
|--------|------------|-------------|
| `agents/` | `context/`, `skill_learning/` | Entry point |
| `context/` | `chromadb`, `sentence-transformers` | All agents |
| `mcp_servers/` | `skill_learning/`, `envs/` | External tools |
| `skill_learning/` | `stable-baselines3`, `envs/` | Training |
| `envs/` | `mujoco`, `gymnasium` | Simulation |
| `explainability/` | `torch`, `captum` | Research agent |

## Entry Points

- **Interactive mode**: `run_orchestrator.py` → `agents/multi_agent_orchestrator.py`
- **MCP servers**: `python -m src.mcp_servers.learning_server`
- **Direct training**: `experiments/mujoco/train.py`

## Conventions

- **Naming**: `snake_case` for files and functions, `PascalCase` for classes
- **Type hints**: Required for all public functions
- **Docstrings**: Google style
- **Logging**: Use `src/utils/conversation_recorder.py` for agent interactions
