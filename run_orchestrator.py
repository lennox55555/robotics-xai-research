#!/usr/bin/env python3
"""
Multi-Agent Robot Skill Learning System

A sophisticated LLM-orchestrated system for teaching humanoid robots new skills.

Architecture:
- Orchestrator: Central coordinator (Claude LLM)
- Learning Agent: RL training & reward design (Claude LLM)
- Performance Agent: Simulation & testing (Claude LLM)
- Research Agent: XAI & analysis (Claude LLM)

Context Engineering:
- RAG: Long-term memory with vector embeddings
- Sliding Window: Recent conversation history
- Handoffs: Structured context passing between agents

Usage:
    python run_orchestrator.py              # Interactive mode
    python run_orchestrator.py --task "..." # Single task
    python run_orchestrator.py --history    # View past sessions
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.multi_agent_orchestrator import create_multi_agent_orchestrator
from src.utils.conversation_recorder import ConversationRecorder


def interactive_mode():
    """Run in interactive chat mode."""
    orchestrator = create_multi_agent_orchestrator()
    orchestrator.run_interactive()


def single_task_mode(task: str):
    """Run a single task and exit."""
    print(f"Processing: {task}\n")
    print("-" * 60)

    orchestrator = create_multi_agent_orchestrator()
    orchestrator.recorder.start_conversation(task[:50])

    response = orchestrator.process_user_input(task)
    print(response)

    # Save conversation
    orchestrator.recorder.set_summary(f"Task: {task}")
    json_path, md_path = orchestrator.recorder.end_conversation()
    print(f"\nConversation saved to: {md_path}")


def show_history():
    """Show past conversation sessions."""
    recorder = ConversationRecorder()
    sessions = recorder.list_conversations()

    if not sessions:
        print("No conversation history found.")
        return

    print("\nPast Conversations\n")
    print("-" * 70)

    for session in sessions[:20]:
        print(f"ID: {session['id']}")
        print(f"   Title: {session['title']}")
        print(f"   Time: {session['start_time']}")
        print(f"   Messages: {session['num_messages']}")
        if session['skills_trained']:
            print(f"   Skills: {', '.join(session['skills_trained'])}")
        print()

    print("-" * 70)
    print(f"\nConversation logs: logs/conversations/")
    print("View a session: cat logs/conversations/<id>.md")


def view_session(session_id: str):
    """View a specific session."""
    recorder = ConversationRecorder()
    conv = recorder.load_conversation(session_id)

    if conv is None:
        print(f"Session not found: {session_id}")
        return

    print(conv.to_markdown())


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Robot Skill Learning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_orchestrator.py
      Start interactive mode

  python run_orchestrator.py --task "teach the robot to walk forward"
      Run a single task

  python run_orchestrator.py --history
      View past sessions

  python run_orchestrator.py --view conv_20240101_120000_abc123
      View a specific session
"""
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Run a single task instead of interactive mode"
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Show past conversation sessions"
    )

    parser.add_argument(
        "--view",
        type=str,
        metavar="SESSION_ID",
        help="View a specific conversation session"
    )

    args = parser.parse_args()

    if args.history:
        show_history()
    elif args.view:
        view_session(args.view)
    elif args.task:
        single_task_mode(args.task)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
