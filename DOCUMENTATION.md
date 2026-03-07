# Documentation Guide

This document explains the documentation structure used throughout this project. When working on this codebase, refer to these files to quickly understand architecture, motivations, and design decisions.

## Documentation File Types

### `designIntentions.md`
**Purpose**: Explains the architectural purpose and design motivations for a directory.

**Location**: Found in each major directory and subdirectory.

**Contains**:
- Why this module/directory exists
- Key design decisions and their rationale
- How it fits into the larger system
- Important patterns and conventions used
- Dependencies and relationships with other modules

**When to read**: Before modifying any code in that directory.

**When to update**: After making significant architectural changes.

---

### `README.md`
**Purpose**: Quick-start and usage documentation.

**Contains**:
- How to run/use the code
- Installation instructions
- Basic examples

---

### `ARCHITECTURE.md` (this project: see below)
**Purpose**: High-level system architecture overview.

**Contains**:
- System diagrams
- Component interactions
- Data flow

---

## Quick Navigation

| Directory | Purpose | Key File |
|-----------|---------|----------|
| `/` | Project root | `DOCUMENTATION.md` (this file) |
| `/src` | All source code | `src/designIntentions.md` |
| `/src/agents` | Multi-agent system | `src/agents/designIntentions.md` |
| `/src/context` | Context engineering (RAG + memory) | `src/context/designIntentions.md` |
| `/src/mcp_servers` | MCP tool servers | `src/mcp_servers/designIntentions.md` |
| `/src/skill_learning` | RL skill training | `src/skill_learning/designIntentions.md` |
| `/src/envs` | Gym environments | `src/envs/designIntentions.md` |
| `/src/explainability` | XAI tools | `src/explainability/designIntentions.md` |
| `/skills` | Trained skill storage | `skills/designIntentions.md` |
| `/logs` | Conversation & training logs | `logs/designIntentions.md` |
| `/configs` | Configuration files | `configs/designIntentions.md` |

---

## For Claude Code / AI Assistants

When starting work on this project:

1. **Read this file first** to understand documentation structure
2. **Read `src/designIntentions.md`** for overall source code architecture
3. **Read the specific `designIntentions.md`** for the module you're modifying
4. **Check `ARCHITECTURE.md`** for system diagrams and data flow

This project uses a **multi-agent architecture** where:
- Each agent has its own LLM instance (Claude)
- Context is shared via RAG + sliding window
- Agents communicate through structured handoffs
- All conversations are logged for reproducibility

---

## Project Summary

**What**: LLM-orchestrated system for teaching humanoid robots new skills

**How**:
- Orchestrator decomposes tasks into learnable skills
- Learning Agent designs RL experiments
- Performance Agent tests in MuJoCo simulation
- Research Agent provides XAI explanations

**Key Innovation**: Hybrid context engineering combining RAG (long-term memory) with sliding window (recent context) for effective multi-agent coordination.
