"""
Message Types for Agent Communication

Defines the structured message format for inter-agent communication.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
import json
import hashlib


class MessageType(Enum):
    """Types of messages in the system."""
    # User interactions
    USER_INPUT = "user_input"
    USER_FEEDBACK = "user_feedback"

    # Agent reasoning
    THOUGHT = "thought"
    PLAN = "plan"
    DECISION = "decision"

    # Actions
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ACTION = "action"

    # Inter-agent
    HANDOFF = "handoff"
    QUERY = "query"
    RESPONSE = "response"
    CONTEXT_SHARE = "context_share"

    # System
    SYSTEM = "system"
    ERROR = "error"
    SUMMARY = "summary"


class AgentRole(Enum):
    """Agent identifiers."""
    USER = "user"
    ORCHESTRATOR = "orchestrator"
    LEARNING = "learning_agent"
    PERFORMANCE = "performance_agent"
    RESEARCH = "research_agent"
    SYSTEM = "system"


@dataclass
class AgentMessage:
    """
    A structured message for agent communication.

    This is the core unit of information passed between agents.
    """

    # Identity
    id: str = field(default_factory=lambda: hashlib.md5(
        f"{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12])

    # Source and destination
    source: str = "system"  # AgentRole value
    destination: Optional[str] = None  # None = broadcast

    # Content
    message_type: str = "thought"  # MessageType value
    content: str = ""

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Context references
    parent_id: Optional[str] = None  # ID of message this responds to
    thread_id: Optional[str] = None  # Conversation thread

    # For tool calls
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[Any] = None

    # Importance for RAG retrieval
    importance: float = 0.5  # 0-1 scale

    # Token count (for budget management)
    token_count: Optional[int] = None

    def __post_init__(self):
        """Estimate token count if not provided."""
        if self.token_count is None:
            # Rough estimate: 1 token ≈ 4 characters
            self.token_count = len(self.content) // 4 + 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "destination": self.destination,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "importance": self.importance,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create from dictionary."""
        return cls(**data)

    def to_llm_format(self) -> Dict[str, str]:
        """Convert to format suitable for LLM context."""
        role_map = {
            "user": "user",
            "orchestrator": "assistant",
            "learning_agent": "assistant",
            "performance_agent": "assistant",
            "research_agent": "assistant",
            "system": "system",
        }

        role = role_map.get(self.source, "assistant")

        # Format content with metadata
        formatted_content = self.content

        if self.tool_name:
            formatted_content = f"[Tool: {self.tool_name}]\n{formatted_content}"

        if self.source not in ["user", "system"]:
            formatted_content = f"[{self.source.upper()}] {formatted_content}"

        return {
            "role": role,
            "content": formatted_content,
        }

    def to_embedding_text(self) -> str:
        """Convert to text for embedding."""
        parts = [
            f"Source: {self.source}",
            f"Type: {self.message_type}",
            f"Content: {self.content}",
        ]

        if self.tool_name:
            parts.append(f"Tool: {self.tool_name}")

        if self.metadata:
            parts.append(f"Metadata: {json.dumps(self.metadata)}")

        return "\n".join(parts)


@dataclass
class HandoffMessage(AgentMessage):
    """
    Specialized message for agent-to-agent handoffs.

    Contains summarized context and specific instructions.
    """

    # Handoff-specific fields
    context_summary: str = ""
    instructions: str = ""
    relevant_history: List[str] = field(default_factory=list)  # Message IDs
    expected_output: str = ""

    def __post_init__(self):
        self.message_type = MessageType.HANDOFF.value
        super().__post_init__()


@dataclass
class ToolCallMessage(AgentMessage):
    """Specialized message for tool calls."""

    def __post_init__(self):
        self.message_type = MessageType.TOOL_CALL.value
        super().__post_init__()


@dataclass
class ToolResultMessage(AgentMessage):
    """Specialized message for tool results."""

    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        self.message_type = MessageType.TOOL_RESULT.value
        super().__post_init__()
