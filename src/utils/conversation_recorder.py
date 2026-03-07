"""
Conversation and Action Recorder

Records the full interaction history in a structured, readable format:
- User natural language inputs
- Agent reasoning and decisions
- Tool calls and results
- Training progress and metrics
- Final outcomes

Outputs both machine-readable JSON and human-readable markdown logs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


class MessageRole(Enum):
    USER = "user"
    ORCHESTRATOR = "orchestrator"
    LEARNING_AGENT = "learning_agent"
    PERFORMANCE_AGENT = "performance_agent"
    RESEARCH_AGENT = "research_agent"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in the conversation."""
    timestamp: str
    role: str
    content: str
    message_type: str  # "text", "action", "tool_call", "tool_result", "thought"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        role_label = {
            "user": "USER",
            "orchestrator": "ORCHESTRATOR",
            "learning_agent": "LEARNING_AGENT",
            "performance_agent": "PERFORMANCE_AGENT",
            "research_agent": "RESEARCH_AGENT",
            "system": "SYSTEM",
        }.get(self.role, self.role.upper())

        type_label = {
            "text": "",
            "action": " [Action]",
            "tool_call": " [Tool Call]",
            "tool_result": " [Result]",
            "thought": " [Thinking]",
        }.get(self.message_type, "")

        header = f"### {role_label}{type_label}\n"
        header += f"*{self.timestamp}*\n\n"

        content = self.content
        if self.metadata:
            content += f"\n\n<details>\n<summary>Details</summary>\n\n```json\n{json.dumps(self.metadata, indent=2)}\n```\n</details>"

        return header + content + "\n\n---\n\n"


@dataclass
class SkillTrainingRecord:
    """Record of a skill training run."""
    skill_id: str
    skill_name: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "pending"  # pending, training, completed, failed
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    transfer_from: Optional[str] = None


@dataclass
class Conversation:
    """A complete conversation/task session."""
    conversation_id: str
    title: str
    start_time: str
    end_time: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    skills_trained: List[SkillTrainingRecord] = field(default_factory=list)
    task_decomposition: Optional[Dict] = None
    summary: Optional[str] = None

    def add_message(
        self,
        role: str,
        content: str,
        message_type: str = "text",
        metadata: Dict = None,
    ) -> Message:
        """Add a message to the conversation."""
        msg = Message(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message("user", content, "text")

    def add_orchestrator_thought(self, content: str) -> Message:
        """Add orchestrator reasoning."""
        return self.add_message("orchestrator", content, "thought")

    def add_orchestrator_action(self, action: str, details: Dict = None) -> Message:
        """Add orchestrator action."""
        return self.add_message("orchestrator", action, "action", details)

    def add_agent_tool_call(
        self,
        agent: str,  # "learning_agent", "performance_agent", "research_agent"
        tool_name: str,
        arguments: Dict,
    ) -> Message:
        """Add an agent tool call."""
        content = f"Calling `{tool_name}`"
        return self.add_message(agent, content, "tool_call", {
            "tool": tool_name,
            "arguments": arguments,
        })

    def add_agent_result(
        self,
        agent: str,
        result: Any,
    ) -> Message:
        """Add an agent tool result."""
        if isinstance(result, dict):
            content = json.dumps(result, indent=2)
        else:
            content = str(result)
        return self.add_message(agent, content, "tool_result")

    def add_training_record(self, record: SkillTrainingRecord):
        """Add a skill training record."""
        self.skills_trained.append(record)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "messages": [asdict(m) for m in self.messages],
            "skills_trained": [asdict(s) for s in self.skills_trained],
            "task_decomposition": self.task_decomposition,
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Convert to readable markdown format."""
        md = f"# Conversation: {self.title}\n\n"
        md += f"**ID**: `{self.conversation_id}`\n"
        md += f"**Started**: {self.start_time}\n"
        if self.end_time:
            md += f"**Ended**: {self.end_time}\n"
        md += "\n---\n\n"

        # Task decomposition
        if self.task_decomposition:
            md += "## Task Decomposition\n\n"
            md += f"**Original Task**: {self.task_decomposition.get('original_prompt', 'N/A')}\n\n"
            md += "**Skills Identified**:\n"
            for skill in self.task_decomposition.get('skills', []):
                md += f"- `{skill.get('skill_id', 'unknown')}`: {skill.get('name', 'Unknown')}\n"
            md += "\n---\n\n"

        # Messages
        md += "## Conversation Log\n\n"
        for msg in self.messages:
            md += msg.to_markdown()

        # Training summary
        if self.skills_trained:
            md += "## Training Summary\n\n"
            md += "| Skill | Status | Reward | Duration |\n"
            md += "|-------|--------|--------|----------|\n"
            for record in self.skills_trained:
                reward = record.metrics.get("mean_reward", "N/A")
                duration = "N/A"
                if record.start_time and record.end_time:
                    # Calculate duration
                    duration = "Completed"
                md += f"| {record.skill_name} | {record.status} | {reward} | {duration} |\n"
            md += "\n"

        # Summary
        if self.summary:
            md += "## Summary\n\n"
            md += self.summary + "\n"

        return md


class ConversationRecorder:
    """
    Records and persists conversations with full agent interaction history.

    Saves both JSON (machine-readable) and Markdown (human-readable) formats.
    """

    def __init__(self, logs_dir: Path = None):
        if logs_dir is None:
            logs_dir = Path(__file__).parent.parent.parent / "logs"
        self.logs_dir = Path(logs_dir)
        self.conversations_dir = self.logs_dir / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self.current: Optional[Conversation] = None

    def start_conversation(self, title: str = None) -> Conversation:
        """Start a new conversation."""
        conv_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self.current = Conversation(
            conversation_id=conv_id,
            title=title or "Untitled Conversation",
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        print(f"Recording conversation: {conv_id}")
        return self.current

    def user_says(self, content: str):
        """Record user input."""
        if self.current is None:
            self.start_conversation(content[:50])
        self.current.add_user_message(content)

    def orchestrator_thinks(self, content: str):
        """Record orchestrator reasoning."""
        if self.current:
            self.current.add_orchestrator_thought(content)

    def orchestrator_acts(self, action: str, details: Dict = None):
        """Record orchestrator action."""
        if self.current:
            self.current.add_orchestrator_action(action, details)

    def orchestrator_says(self, content: str):
        """Record orchestrator response to user."""
        if self.current:
            self.current.add_message("orchestrator", content, "text")

    def learning_agent_calls(self, tool: str, args: Dict):
        """Record learning agent tool call."""
        if self.current:
            self.current.add_agent_tool_call("learning_agent", tool, args)

    def learning_agent_returns(self, result: Any):
        """Record learning agent result."""
        if self.current:
            self.current.add_agent_result("learning_agent", result)

    def performance_agent_calls(self, tool: str, args: Dict):
        """Record performance agent tool call."""
        if self.current:
            self.current.add_agent_tool_call("performance_agent", tool, args)

    def performance_agent_returns(self, result: Any):
        """Record performance agent result."""
        if self.current:
            self.current.add_agent_result("performance_agent", result)

    def research_agent_calls(self, tool: str, args: Dict):
        """Record research agent tool call."""
        if self.current:
            self.current.add_agent_tool_call("research_agent", tool, args)

    def research_agent_returns(self, result: Any):
        """Record research agent result."""
        if self.current:
            self.current.add_agent_result("research_agent", result)

    def record_skill_training(
        self,
        skill_id: str,
        skill_name: str,
        config: Dict = None,
        transfer_from: str = None,
    ) -> SkillTrainingRecord:
        """Start recording a skill training run."""
        record = SkillTrainingRecord(
            skill_id=skill_id,
            skill_name=skill_name,
            start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="training",
            config=config or {},
            transfer_from=transfer_from,
        )
        if self.current:
            self.current.add_training_record(record)
        return record

    def complete_skill_training(
        self,
        record: SkillTrainingRecord,
        metrics: Dict,
        success: bool = True,
    ):
        """Complete a skill training record."""
        record.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.status = "completed" if success else "failed"
        record.metrics = metrics

    def set_task_decomposition(self, decomposition: Dict):
        """Set the task decomposition."""
        if self.current:
            self.current.task_decomposition = decomposition

    def set_summary(self, summary: str):
        """Set the conversation summary."""
        if self.current:
            self.current.summary = summary

    def end_conversation(self) -> tuple[Path, Path]:
        """End and save the current conversation."""
        if self.current is None:
            return None, None

        self.current.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save JSON
        json_path = self.conversations_dir / f"{self.current.conversation_id}.json"
        with open(json_path, "w") as f:
            json.dump(self.current.to_dict(), f, indent=2)

        # Save Markdown
        md_path = self.conversations_dir / f"{self.current.conversation_id}.md"
        with open(md_path, "w") as f:
            f.write(self.current.to_markdown())

        print(f"Conversation saved:")
        print(f"   JSON: {json_path}")
        print(f"   Markdown: {md_path}")

        conv = self.current
        self.current = None

        return json_path, md_path

    def list_conversations(self) -> List[Dict]:
        """List all recorded conversations."""
        conversations = []

        for json_file in sorted(self.conversations_dir.glob("*.json"), reverse=True):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                conversations.append({
                    "id": data["conversation_id"],
                    "title": data["title"],
                    "start_time": data["start_time"],
                    "num_messages": len(data.get("messages", [])),
                    "skills_trained": len(data.get("skills_trained", [])),
                })
            except Exception:
                continue

        return conversations

    def load_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Load a previous conversation."""
        json_path = self.conversations_dir / f"{conv_id}.json"

        if not json_path.exists():
            return None

        with open(json_path, "r") as f:
            data = json.load(f)

        conv = Conversation(
            conversation_id=data["conversation_id"],
            title=data["title"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            task_decomposition=data.get("task_decomposition"),
            summary=data.get("summary"),
        )

        for msg_data in data.get("messages", []):
            conv.messages.append(Message(**msg_data))

        for skill_data in data.get("skills_trained", []):
            conv.skills_trained.append(SkillTrainingRecord(**skill_data))

        return conv


# Global recorder
_recorder: Optional[ConversationRecorder] = None


def get_conversation_recorder() -> ConversationRecorder:
    """Get the global conversation recorder."""
    global _recorder
    if _recorder is None:
        _recorder = ConversationRecorder()
    return _recorder
