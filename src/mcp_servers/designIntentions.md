# MCP Servers Design Intentions

## Purpose

The `mcp_servers/` module exposes agent capabilities via **Model Context Protocol (MCP)**. This allows agents to call tools and enables external systems to interact with the skill learning infrastructure.

## What is MCP?

MCP (Model Context Protocol) is a standardized way to expose tools to LLMs:
- Tools are defined with JSON schemas
- LLMs can call tools by name with arguments
- Results are returned in a standard format

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT (Claude LLM)                      │
│                                                             │
│  "I need to train the walking skill"                        │
│           │                                                 │
│           ▼                                                 │
│  Tool call: train_skill(skill_id="walk_forward")            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP SERVER                                │
│                                                             │
│  @server.call_tool()                                        │
│  async def call_tool(name, arguments):                      │
│      if name == "train_skill":                              │
│          return trainer.train(arguments["skill_id"])        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 ACTUAL IMPLEMENTATION                        │
│                                                             │
│  SkillTrainer, G1HumanoidEnv, PolicyAnalyzer, etc.          │
└─────────────────────────────────────────────────────────────┘
```

## Server Types

### `learning_server.py` - Learning Agent Tools
| Tool | Purpose |
|------|---------|
| `create_skill` | Define a new skill with reward function |
| `train_skill` | Start RL training |
| `get_training_status` | Check training progress |
| `evaluate_skill` | Evaluate trained skill |
| `suggest_transfer_source` | Find skills for transfer learning |
| `get_reward_components` | List available reward components |

### `performance_server.py` - Performance Agent Tools
| Tool | Purpose |
|------|---------|
| `reset_simulation` | Reset robot to initial state |
| `step_simulation` | Step physics forward |
| `execute_skill` | Run a trained skill |
| `run_skill_episode` | Run full episode |
| `get_robot_state` | Get current sensor readings |
| `get_joint_info` | Get robot joint information |

### `research_server.py` - Research Agent Tools
| Tool | Purpose |
|------|---------|
| `analyze_skill_performance` | Detailed metrics analysis |
| `explain_policy_decision` | XAI explanations |
| `compare_skills` | Compare two skills |
| `analyze_failure_modes` | Failure analysis |
| `suggest_improvements` | Optimization recommendations |
| `generate_skill_report` | Comprehensive report |

## Design Decisions

### Why MCP Instead of Direct Function Calls?

**Benefits**:
1. **Standardization**: Same interface for all tools
2. **Discoverability**: Tools are self-describing with schemas
3. **External Access**: Can be used by other systems
4. **Logging**: All tool calls can be logged automatically

### Why Separate Servers per Agent?

**Problem**: Monolithic server with all tools is hard to maintain.

**Solution**: Each agent has its own server:
- Clear ownership
- Independent deployment
- Focused tool sets

### Why Async?

**Problem**: Training can take minutes/hours.

**Solution**: Async servers allow:
- Non-blocking tool calls
- Progress monitoring
- Concurrent operations

## Running Servers

```bash
# Start individual servers
python -m src.mcp_servers.learning_server
python -m src.mcp_servers.performance_server
python -m src.mcp_servers.research_server
```

## Tool Definition Pattern

```python
@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="train_skill",
            description="Start training a skill",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "config": {"type": "object"},
                },
                "required": ["skill_id"],
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict) -> List[TextContent]:
    if name == "train_skill":
        result = trainer.train(arguments["skill_id"])
        return [TextContent(type="text", text=json.dumps(result))]
```

## Future Extensions

- **Streaming**: Stream training progress updates
- **Authentication**: Secure external access
- **Rate limiting**: Prevent resource exhaustion
- **Webhooks**: Notify on training completion
