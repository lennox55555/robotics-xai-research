"""
Logging and Session Recording

Records all interactions, commands, and training sessions for:
- Research reproducibility
- Debugging
- Analysis of orchestrator decisions
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import uuid


# Configure logging
def setup_logging(log_dir: Path = None, level: int = logging.INFO):
    """Set up logging to both file and console."""
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / "orchestrator" / f"session_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return log_file


@dataclass
class Command:
    """A single command/interaction."""
    timestamp: str
    command_type: str  # "user_input", "agent_action", "tool_call", "response"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """A complete interaction session."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    commands: List[Command] = field(default_factory=list)
    task_prompt: Optional[str] = None
    skills_trained: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_command(
        self,
        command_type: str,
        content: str,
        metadata: Dict[str, Any] = None,
    ):
        """Add a command to the session."""
        cmd = Command(
            timestamp=datetime.now().isoformat(),
            command_type=command_type,
            content=content,
            metadata=metadata or {},
        )
        self.commands.append(cmd)
        return cmd

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "task_prompt": self.task_prompt,
            "skills_trained": self.skills_trained,
            "commands": [asdict(c) for c in self.commands],
            "metadata": self.metadata,
        }


class SessionRecorder:
    """
    Records and persists interaction sessions.

    Usage:
        recorder = SessionRecorder()
        recorder.start_session()
        recorder.log_user_input("Walk forward and jump")
        recorder.log_agent_action("decompose_task", {"skills": ["walk", "jump"]})
        recorder.log_tool_call("train_skill", {"skill_id": "walk"})
        recorder.end_session()
    """

    def __init__(self, sessions_dir: Path = None):
        if sessions_dir is None:
            sessions_dir = Path(__file__).parent.parent.parent / "logs" / "sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[Session] = None
        self.logger = logging.getLogger("SessionRecorder")

    def start_session(self, task_prompt: str = None) -> Session:
        """Start a new recording session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self.current_session = Session(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            task_prompt=task_prompt,
        )

        self.logger.info(f"Started session: {session_id}")
        return self.current_session

    def log_user_input(self, content: str, metadata: Dict = None):
        """Log a user input/command."""
        if self.current_session is None:
            self.start_session()

        self.current_session.add_command("user_input", content, metadata)
        self.logger.info(f"User: {content[:100]}...")

    def log_agent_action(self, action: str, details: Dict = None):
        """Log an agent action/decision."""
        if self.current_session is None:
            return

        self.current_session.add_command(
            "agent_action",
            action,
            {"details": details or {}},
        )
        self.logger.info(f"Agent action: {action}")

    def log_tool_call(self, tool_name: str, arguments: Dict = None, result: Any = None):
        """Log an MCP tool call."""
        if self.current_session is None:
            return

        self.current_session.add_command(
            "tool_call",
            tool_name,
            {
                "arguments": arguments or {},
                "result": str(result)[:500] if result else None,
            },
        )
        self.logger.info(f"Tool call: {tool_name}")

    def log_response(self, content: str, metadata: Dict = None):
        """Log an agent response."""
        if self.current_session is None:
            return

        self.current_session.add_command("response", content, metadata)
        self.logger.debug(f"Response: {content[:100]}...")

    def log_skill_trained(self, skill_id: str, metrics: Dict = None):
        """Log that a skill was trained."""
        if self.current_session is None:
            return

        self.current_session.skills_trained.append(skill_id)
        self.current_session.add_command(
            "skill_trained",
            skill_id,
            {"metrics": metrics or {}},
        )
        self.logger.info(f"Skill trained: {skill_id}")

    def end_session(self) -> Optional[Path]:
        """End the current session and save to disk."""
        if self.current_session is None:
            return None

        self.current_session.end_time = datetime.now().isoformat()

        # Save session
        session_file = self.sessions_dir / f"{self.current_session.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

        self.logger.info(f"Session saved: {session_file}")

        session = self.current_session
        self.current_session = None

        return session_file

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a previous session."""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, "r") as f:
            data = json.load(f)

        session = Session(
            session_id=data["session_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            task_prompt=data.get("task_prompt"),
            skills_trained=data.get("skills_trained", []),
            metadata=data.get("metadata", {}),
        )

        for cmd_data in data.get("commands", []):
            session.commands.append(Command(**cmd_data))

        return session

    def list_sessions(self) -> List[Dict]:
        """List all recorded sessions."""
        sessions = []

        for session_file in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "start_time": data["start_time"],
                    "task_prompt": data.get("task_prompt", "")[:50],
                    "num_commands": len(data.get("commands", [])),
                    "skills_trained": data.get("skills_trained", []),
                })
            except Exception:
                continue

        return sessions


# Global recorder instance
_recorder: Optional[SessionRecorder] = None


def get_recorder() -> SessionRecorder:
    """Get the global session recorder."""
    global _recorder
    if _recorder is None:
        _recorder = SessionRecorder()
    return _recorder
