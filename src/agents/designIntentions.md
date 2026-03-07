# Agents Module Design Intentions

## Purpose

The `agents/` module implements a **multi-agent system** where specialized AI agents collaborate to teach a humanoid robot new skills. Each agent is an expert in a specific domain.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                              │
│         (Coordinates, decomposes tasks, synthesizes)         │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   LEARNING    │ │  PERFORMANCE  │ │   RESEARCH    │
│    AGENT      │ │     AGENT     │ │    AGENT      │
│               │ │               │ │               │
│ Trains skills │ │ Runs sims     │ │ Analyzes &    │
│ via RL        │ │ Tests skills  │ │ explains      │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Design Decisions

### Why Multiple Agents?

**Problem**: A single LLM trying to do everything (train, test, analyze) produces mediocre results in each area.

**Solution**: Specialized agents with focused system prompts perform significantly better.

**Evidence**: Research on role-specialized prompting (e.g., "You are an expert in X") shows improved task performance.

### Why Each Agent Has Its Own LLM Instance?

**Problem**: Shared context leads to confusion. Training context pollutes analysis context.

**Solution**: Each agent maintains:
- Own Claude API client
- Own conversation history
- Own system prompt
- Access to shared RAG memory (when relevant)

### Why Structured Handoffs?

**Problem**: Dumping full conversation history to agents wastes tokens and dilutes focus.

**Solution**: Orchestrator creates structured handoffs:
```
HANDOFF TO: learning_agent

TASK: Design a walking skill

CONTEXT:
- User wants robot to walk forward
- No existing walking skill
- Balance skill already trained

EXPECTED OUTPUT:
- Skill definition with reward function
- Training configuration
```

## Agent Responsibilities

### Learning Agent (`base_agent.py::LearningAgent`)
- **Expertise**: Reinforcement learning, reward engineering
- **Tasks**:
  - Decompose skills into sub-skills
  - Design reward functions
  - Configure training hyperparameters
  - Apply transfer learning
- **System Prompt Focus**: RL best practices, reward shaping, curriculum learning

### Performance Agent (`base_agent.py::PerformanceAgent`)
- **Expertise**: Simulation, execution, testing
- **Tasks**:
  - Run trained policies in MuJoCo
  - Collect performance metrics
  - Test robustness (perturbations)
  - Report observations
- **System Prompt Focus**: Testing methodology, metric collection

### Research Agent (`base_agent.py::ResearchAgent`)
- **Expertise**: XAI, analysis, debugging
- **Tasks**:
  - Explain policy decisions (saliency, feature importance)
  - Analyze failure modes
  - Compare training approaches
  - Suggest improvements
- **System Prompt Focus**: Interpretability methods, scientific analysis

## Key Files

| File | Purpose |
|------|---------|
| `base_agent.py` | Base class + all three agent implementations with system prompts |
| `multi_agent_orchestrator.py` | Main coordinator that routes to agents |
| `orchestrator/orchestrator_agent.py` | Legacy single-orchestrator (deprecated) |

## Context Flow

1. User input → Orchestrator
2. Orchestrator plans → Creates handoff
3. Handoff → Target agent (with context summary)
4. Agent thinks → Returns result
5. Result → Orchestrator synthesizes
6. (Optional) Orchestrator → Another agent
7. Final response → User

## Extending

To add a new agent:

1. Create class inheriting `BaseAgent` in `base_agent.py`
2. Define specialized system prompt
3. Define agent-specific tools
4. Register in `multi_agent_orchestrator.py`
5. Add MCP server in `mcp_servers/`
