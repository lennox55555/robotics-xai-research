# Context Engineering Design Intentions

## Purpose

The `context/` module implements **hybrid context management** for the multi-agent system. It solves the fundamental problem of giving LLM agents the right information at the right time.

## The Problem

LLMs have limited context windows. In a multi-agent system:
- Conversations can span hundreds of messages
- Important information from hours ago may be relevant now
- Each agent needs different context
- Raw conversation dumps are inefficient

## The Solution: Hybrid RAG + Sliding Window

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT FOR AGENT                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │           RAG RETRIEVAL (Long-term)                 │    │
│  │                                                     │    │
│  │  Query: "walking skill"                             │    │
│  │    → Retrieved: Past decisions about locomotion    │    │
│  │    → Retrieved: Previous reward function designs   │    │
│  │    → Retrieved: Related failure analyses           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           SLIDING WINDOW (Short-term)               │    │
│  │                                                     │    │
│  │  Last 50 messages in conversation                   │    │
│  │  Maintains flow and immediate context               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           HANDOFF CONTEXT (If present)              │    │
│  │                                                     │    │
│  │  Instructions from orchestrator                     │    │
│  │  Summarized relevant history                        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why RAG?

**Problem**: Important context from earlier in conversation is lost.

**Solution**: Store all messages in vector database, retrieve semantically similar ones.

**Implementation**: ChromaDB + sentence-transformers embeddings.

### Why Sliding Window?

**Problem**: RAG alone misses conversational flow. "What did you just say?" fails.

**Solution**: Always include last N messages regardless of semantic similarity.

### Why Per-Agent Context?

**Problem**: Learning agent doesn't need simulation details. Research agent doesn't need training configs.

**Solution**: Each agent has:
- Own sliding window
- Own relevance keywords for RAG filtering
- Own category filters

### Why Structured Messages?

**Problem**: Raw text loses metadata (who said it, when, importance).

**Solution**: `AgentMessage` dataclass with:
- Source agent
- Message type (thought, action, tool_call, etc.)
- Timestamp
- Importance score (for RAG prioritization)
- Token count (for budget management)

## Key Components

### `message_types.py`
Defines structured message format:
```python
@dataclass
class AgentMessage:
    source: str           # "orchestrator", "learning_agent", etc.
    message_type: str     # "thought", "tool_call", "response"
    content: str
    importance: float     # 0-1, affects RAG retrieval
    token_count: int      # For budget management
```

### `memory_store.py`
Vector database for long-term memory:
- Uses ChromaDB for persistence
- Sentence-transformers for embeddings
- Fallback to in-memory if ChromaDB unavailable
- Semantic search with filters (category, source, importance)

### `context_manager.py`
Orchestrates context building:
- Maintains per-agent sliding windows
- Queries RAG for relevant memories
- Builds handoff messages
- Manages token budgets

## Token Budget Management

Each agent has a token budget (default 6000):
```
Budget allocation:
- Handoff instructions: ~500 tokens (highest priority)
- Task context: ~300 tokens
- RAG memories: ~2000 tokens
- Recent messages: Remaining tokens
```

## Usage Example

```python
# Add a message
context_manager.add_message(
    AgentMessage(
        source="learning_agent",
        message_type="decision",
        content="Using PPO with lr=3e-4 for walking skill",
        importance=0.8,  # High importance → persists in RAG
    ),
    memory_category="training",
)

# Get context for an agent
context = context_manager.get_context_for_agent(
    agent_id="research_agent",
    query="Why did the walking skill fail?",
    include_rag=True,
    include_recent=True,
)
```

## Research Background

This design is influenced by:
- **RAG (Lewis et al., 2020)**: Retrieval-Augmented Generation
- **MemGPT (Packer et al., 2023)**: Tiered memory for LLMs
- **Reflexion (Shinn et al., 2023)**: Episodic memory for agents
