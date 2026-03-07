"""
Context Manager

Hybrid context engineering combining:
1. RAG (Retrieval-Augmented Generation) for long-term memory
2. Sliding window for recent conversation history
3. Token budget management
4. Cross-agent context sharing
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import json

from src.context.message_types import AgentMessage, MessageType, AgentRole
from src.context.memory_store import MemoryStore, Memory


@dataclass
class ContextWindow:
    """
    A sliding window of recent messages.

    Maintains the most recent N messages or T tokens.
    """

    max_messages: int = 50
    max_tokens: int = 8000
    messages: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, message: AgentMessage):
        """Add a message to the window."""
        self.messages.append(message)
        self._trim_to_budget()

    def _trim_to_budget(self):
        """Trim messages to fit token budget."""
        total_tokens = sum(m.token_count or 0 for m in self.messages)

        while total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.popleft()
            total_tokens -= removed.token_count or 0

    def get_recent(self, n: int = None) -> List[AgentMessage]:
        """Get the N most recent messages."""
        if n is None:
            n = self.max_messages
        return list(self.messages)[-n:]

    def get_by_source(self, source: str) -> List[AgentMessage]:
        """Get messages from a specific source."""
        return [m for m in self.messages if m.source == source]

    def get_by_type(self, message_type: str) -> List[AgentMessage]:
        """Get messages of a specific type."""
        return [m for m in self.messages if m.message_type == message_type]

    def to_llm_messages(self, n: int = None) -> List[Dict[str, str]]:
        """Convert to LLM message format."""
        messages = self.get_recent(n)
        return [m.to_llm_format() for m in messages]

    def clear(self):
        """Clear the window."""
        self.messages.clear()

    @property
    def token_count(self) -> int:
        """Get total token count."""
        return sum(m.token_count or 0 for m in self.messages)


@dataclass
class AgentContext:
    """
    Context specific to an agent.

    Each agent maintains its own context window and can access shared memory.
    """

    agent_id: str
    window: ContextWindow = field(default_factory=ContextWindow)

    # Agent-specific memory categories
    categories: List[str] = field(default_factory=list)

    # What this agent cares about
    relevance_keywords: List[str] = field(default_factory=list)

    # Last handoff received
    last_handoff: Optional[AgentMessage] = None

    # Current task context
    current_task: Optional[str] = None
    current_skill: Optional[str] = None


class ContextManager:
    """
    Central context management for the multi-agent system.

    Provides:
    - Per-agent context windows (recent messages)
    - Shared long-term memory (RAG)
    - Cross-agent context sharing
    - Token budget management
    - Context summarization
    """

    def __init__(
        self,
        persist_dir: Path = None,
        max_tokens_per_agent: int = 6000,
        max_rag_results: int = 5,
        rag_token_budget: int = 2000,
    ):
        # Memory store for RAG
        self.memory_store = MemoryStore(
            persist_dir=persist_dir / "memory" if persist_dir else None,
        )

        # Per-agent contexts
        self.agent_contexts: Dict[str, AgentContext] = {}

        # Global conversation history
        self.global_history: List[AgentMessage] = []

        # Configuration
        self.max_tokens_per_agent = max_tokens_per_agent
        self.max_rag_results = max_rag_results
        self.rag_token_budget = rag_token_budget

        # Initialize agent contexts
        self._init_agent_contexts()

    def _init_agent_contexts(self):
        """Initialize context for each agent."""
        agents = {
            "orchestrator": AgentContext(
                agent_id="orchestrator",
                categories=["task", "decision", "handoff"],
                relevance_keywords=["task", "skill", "plan", "coordinate"],
            ),
            "learning_agent": AgentContext(
                agent_id="learning_agent",
                categories=["training", "skill", "model"],
                relevance_keywords=["train", "learn", "model", "reward", "policy"],
            ),
            "performance_agent": AgentContext(
                agent_id="performance_agent",
                categories=["simulation", "execution", "observation"],
                relevance_keywords=["run", "execute", "simulate", "robot", "action"],
            ),
            "research_agent": AgentContext(
                agent_id="research_agent",
                categories=["analysis", "explanation", "insight"],
                relevance_keywords=["analyze", "explain", "compare", "improve", "xai"],
            ),
        }

        for agent_id, context in agents.items():
            self.agent_contexts[agent_id] = context

    def add_message(
        self,
        message: AgentMessage,
        store_in_memory: bool = True,
        memory_category: str = "general",
    ):
        """
        Add a message to the context system.

        - Adds to global history
        - Adds to relevant agent windows
        - Optionally stores in long-term memory
        """
        # Add to global history
        self.global_history.append(message)

        # Add to source agent's window
        if message.source in self.agent_contexts:
            self.agent_contexts[message.source].window.add(message)

        # Add to destination agent's window if specified
        if message.destination and message.destination in self.agent_contexts:
            self.agent_contexts[message.destination].window.add(message)

        # Broadcast important messages to all agents
        if message.importance >= 0.7:
            for agent_id, context in self.agent_contexts.items():
                if agent_id not in [message.source, message.destination]:
                    context.window.add(message)

        # Store in long-term memory
        if store_in_memory and message.importance >= 0.3:
            self.memory_store.add_message(message, category=memory_category)

    def get_context_for_agent(
        self,
        agent_id: str,
        query: str = None,
        include_rag: bool = True,
        include_recent: bool = True,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Build context for an agent's LLM call.

        Combines:
        1. Recent messages from sliding window
        2. Relevant memories from RAG
        3. Current task context
        4. Last handoff instructions
        """
        if max_tokens is None:
            max_tokens = self.max_tokens_per_agent

        context = {
            "recent_messages": [],
            "rag_memories": [],
            "task_context": None,
            "handoff": None,
            "token_count": 0,
        }

        agent_context = self.agent_contexts.get(agent_id)
        if not agent_context:
            return context

        token_budget = max_tokens

        # 1. Add handoff context if present (highest priority)
        if agent_context.last_handoff:
            context["handoff"] = agent_context.last_handoff.to_llm_format()
            token_budget -= agent_context.last_handoff.token_count or 200

        # 2. Add task context
        if agent_context.current_task:
            context["task_context"] = agent_context.current_task
            token_budget -= len(agent_context.current_task) // 4

        # 3. Add RAG memories if requested
        if include_rag and query:
            rag_budget = min(self.rag_token_budget, token_budget // 3)
            memories = self._retrieve_memories(
                query=query,
                agent_id=agent_id,
                token_budget=rag_budget,
            )
            context["rag_memories"] = memories
            token_budget -= sum(len(m["content"]) // 4 for m in memories)

        # 4. Add recent messages
        if include_recent:
            recent = agent_context.window.get_recent()

            # Fit within budget
            recent_messages = []
            for msg in reversed(recent):
                if token_budget <= 0:
                    break
                msg_tokens = msg.token_count or 0
                if msg_tokens <= token_budget:
                    recent_messages.insert(0, msg.to_llm_format())
                    token_budget -= msg_tokens

            context["recent_messages"] = recent_messages

        context["token_count"] = max_tokens - token_budget
        return context

    def _retrieve_memories(
        self,
        query: str,
        agent_id: str,
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from RAG."""
        agent_context = self.agent_contexts.get(agent_id)
        if not agent_context:
            return []

        # Search with agent's relevance categories
        results = self.memory_store.search(
            query=query,
            n_results=self.max_rag_results * 2,
            min_importance=0.3,
        )

        # Filter and format
        memories = []
        tokens_used = 0

        for memory, score in results:
            mem_tokens = len(memory.content) // 4

            if tokens_used + mem_tokens > token_budget:
                continue

            memories.append({
                "content": memory.content,
                "source": memory.source_agent,
                "category": memory.category,
                "relevance": score,
                "timestamp": memory.timestamp,
            })

            tokens_used += mem_tokens

            if len(memories) >= self.max_rag_results:
                break

        return memories

    def handoff_to_agent(
        self,
        from_agent: str,
        to_agent: str,
        instructions: str,
        context_summary: str,
        relevant_messages: List[AgentMessage] = None,
    ) -> AgentMessage:
        """
        Create a handoff from one agent to another.

        Packages relevant context and instructions for the receiving agent.
        """
        # Get recent context from source agent
        source_context = self.agent_contexts.get(from_agent)
        recent = source_context.window.get_recent(10) if source_context else []

        # Build handoff message
        handoff = AgentMessage(
            source=from_agent,
            destination=to_agent,
            message_type=MessageType.HANDOFF.value,
            content=instructions,
            metadata={
                "context_summary": context_summary,
                "source_recent": [m.to_dict() for m in recent[-5:]],
            },
            importance=0.9,
        )

        # Store handoff in destination agent's context
        if to_agent in self.agent_contexts:
            self.agent_contexts[to_agent].last_handoff = handoff

        self.add_message(handoff, memory_category="handoff")

        return handoff

    def set_agent_task(self, agent_id: str, task: str, skill: str = None):
        """Set the current task context for an agent."""
        if agent_id in self.agent_contexts:
            self.agent_contexts[agent_id].current_task = task
            self.agent_contexts[agent_id].current_skill = skill

    def clear_agent_handoff(self, agent_id: str):
        """Clear handoff after agent has processed it."""
        if agent_id in self.agent_contexts:
            self.agent_contexts[agent_id].last_handoff = None

    def build_system_context(self, agent_id: str) -> str:
        """
        Build system context string for an agent.

        Includes agent-specific instructions and current state.
        """
        agent_context = self.agent_contexts.get(agent_id)
        if not agent_context:
            return ""

        parts = []

        # Current task
        if agent_context.current_task:
            parts.append(f"## Current Task\n{agent_context.current_task}")

        # Current skill
        if agent_context.current_skill:
            parts.append(f"## Current Skill\n{agent_context.current_skill}")

        # Handoff instructions
        if agent_context.last_handoff:
            parts.append(f"## Instructions from {agent_context.last_handoff.source}")
            parts.append(agent_context.last_handoff.content)
            if agent_context.last_handoff.metadata.get("context_summary"):
                parts.append(f"\n### Context\n{agent_context.last_handoff.metadata['context_summary']}")

        return "\n\n".join(parts)

    def summarize_conversation(self, last_n: int = 20) -> str:
        """Create a summary of recent conversation."""
        recent = self.global_history[-last_n:]

        if not recent:
            return "No conversation history."

        summary_parts = []

        # Group by type
        user_inputs = [m for m in recent if m.source == "user"]
        decisions = [m for m in recent if m.message_type == "decision"]
        tool_calls = [m for m in recent if m.message_type == "tool_call"]

        if user_inputs:
            summary_parts.append("User requests:")
            for m in user_inputs[-3:]:
                summary_parts.append(f"  - {m.content[:100]}")

        if decisions:
            summary_parts.append("Decisions made:")
            for m in decisions[-3:]:
                summary_parts.append(f"  - {m.content[:100]}")

        if tool_calls:
            summary_parts.append("Actions taken:")
            for m in tool_calls[-5:]:
                summary_parts.append(f"  - {m.tool_name}: {m.content[:50]}")

        return "\n".join(summary_parts)

    def get_conversation_for_logging(self) -> List[Dict]:
        """Get full conversation in loggable format."""
        return [m.to_dict() for m in self.global_history]

    def save_state(self, path: Path):
        """Save context manager state to disk."""
        state = {
            "global_history": [m.to_dict() for m in self.global_history],
            "agent_states": {
                agent_id: {
                    "current_task": ctx.current_task,
                    "current_skill": ctx.current_skill,
                    "window_messages": [m.to_dict() for m in ctx.window.messages],
                }
                for agent_id, ctx in self.agent_contexts.items()
            },
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path):
        """Load context manager state from disk."""
        if not path.exists():
            return

        with open(path, "r") as f:
            state = json.load(f)

        self.global_history = [
            AgentMessage.from_dict(m) for m in state.get("global_history", [])
        ]

        for agent_id, agent_state in state.get("agent_states", {}).items():
            if agent_id in self.agent_contexts:
                ctx = self.agent_contexts[agent_id]
                ctx.current_task = agent_state.get("current_task")
                ctx.current_skill = agent_state.get("current_skill")

                for msg_data in agent_state.get("window_messages", []):
                    ctx.window.add(AgentMessage.from_dict(msg_data))
