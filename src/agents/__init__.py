"""
Agent System for Robotics Skill Learning

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│              (Main brain - LangGraph-based)                  │
│                                                             │
│   Receives tasks → Decomposes → Coordinates → Reports      │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   LEARNING    │ │  PERFORMANCE  │ │   RESEARCH    │
│    AGENT      │ │     AGENT     │ │    AGENT      │
│               │ │               │ │               │
│ - Train skills│ │ - Run sims    │ │ - Analyze     │
│ - Transfer    │ │ - Execute     │ │ - Explain     │
│ - Evaluate    │ │ - Test        │ │ - Compare     │
│               │ │               │ │               │
│  MCP Server   │ │  MCP Server   │ │  MCP Server   │
└───────────────┘ └───────────────┘ └───────────────┘
"""

from src.agents.orchestrator.orchestrator_agent import (
    OrchestratorAgent,
    create_orchestrator,
)

__all__ = [
    "OrchestratorAgent",
    "create_orchestrator",
]
