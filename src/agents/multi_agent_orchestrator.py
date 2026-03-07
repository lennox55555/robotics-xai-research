"""
Multi-Agent Orchestrator

Coordinates the Learning, Performance, and Research agents using:
- LangGraph for state machine orchestration
- Hybrid RAG + sliding window context
- Structured handoffs between agents
- Full conversation logging

Architecture based on:
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
- "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

from dotenv import load_dotenv
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context.context_manager import ContextManager
from src.context.message_types import AgentMessage, MessageType, AgentRole
from src.agents.base_agent import LearningAgent, PerformanceAgent, ResearchAgent, AgentConfig
from src.utils.conversation_recorder import ConversationRecorder, get_conversation_recorder
from src.skill_learning.skill import SkillLibrary

load_dotenv()


# =============================================================================
# ORCHESTRATOR SYSTEM PROMPT
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are the **Orchestrator** for an advanced humanoid robot skill learning system.

## Your Role
You are the central coordinator that:
1. Understands user requests in natural language
2. Decomposes complex tasks into learnable skills
3. Delegates work to specialized agents
4. Synthesizes results into coherent responses
5. Maintains conversation context and memory

## Your Team

### Learning Agent
**Expertise**: Reinforcement learning, reward design, training
**Use when**:
- Creating new skill definitions
- Setting up training experiments
- Designing reward functions
- Applying transfer learning

### Performance Agent
**Expertise**: Simulation, execution, testing
**Use when**:
- Testing trained skills
- Running simulations
- Collecting performance data
- Evaluating robustness

### Research Agent
**Expertise**: Analysis, XAI, insights
**Use when**:
- Explaining policy decisions
- Analyzing failures
- Comparing approaches
- Generating insights

## Your Workflow

When a user asks something like "teach the robot to walk and jump":

1. **Understand**: Parse the request, identify the high-level goal
2. **Decompose**: Break into skills: balance → walk → jump
3. **Plan**: Determine order, dependencies, transfer opportunities
4. **Delegate**:
   - Learning Agent: Create skill definitions, start training
   - Performance Agent: Test each skill
   - Research Agent: Analyze results, suggest improvements
5. **Iterate**: Refine based on feedback
6. **Report**: Summarize progress to user

## Handoff Protocol

When delegating to an agent, provide:
```
HANDOFF TO: [agent_name]

TASK: [specific task]

CONTEXT:
- User's original request: [...]
- Current progress: [...]
- Relevant history: [...]

EXPECTED OUTPUT:
- [what you need back]

CONSTRAINTS:
- [any limitations or requirements]
```

## Decision Making

**Route to Learning Agent when**:
- User mentions: train, learn, teach, create skill, reward
- Need to: design experiments, set hyperparameters, transfer learn

**Route to Performance Agent when**:
- User mentions: test, run, execute, simulate, show
- Need to: validate skills, collect data, demonstrate

**Route to Research Agent when**:
- User mentions: why, explain, analyze, compare, improve
- Need to: debug, understand, optimize

## Important Guidelines

1. Always acknowledge user requests first
2. Explain your plan before executing
3. Keep users informed of progress
4. Synthesize agent responses into clear summaries
5. Ask clarifying questions when needed
6. Learn from failures and adjust approach
"""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class OrchestratorState(TypedDict):
    """State for the orchestrator state machine."""
    # Conversation
    messages: Annotated[List[Dict], operator.add]
    user_request: str

    # Task tracking
    task_id: str
    task_decomposition: Optional[Dict]
    current_phase: str  # understand, plan, execute, analyze, report

    # Skill tracking
    skills_to_train: List[str]
    skills_in_progress: List[str]
    skills_completed: List[str]

    # Agent coordination
    active_agent: Optional[str]
    pending_handoffs: List[Dict]
    agent_results: Dict[str, Any]

    # Iteration
    iteration: int
    max_iterations: int


# =============================================================================
# MULTI-AGENT ORCHESTRATOR
# =============================================================================

class MultiAgentOrchestrator:
    """
    Coordinates multiple specialized agents for robot skill learning.

    Features:
    - Each agent has its own LLM instance
    - Hybrid context (RAG + sliding window)
    - Structured handoffs with context summaries
    - Full conversation logging
    - Iterative refinement
    """

    def __init__(
        self,
        persist_dir: Path = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        if persist_dir is None:
            persist_dir = Path(__file__).parent.parent.parent / "data"

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize context manager (shared across agents)
        self.context_manager = ContextManager(
            persist_dir=self.persist_dir / "context",
            max_tokens_per_agent=6000,
            max_rag_results=5,
        )

        # Initialize skill library
        self.skill_library = SkillLibrary(
            self.persist_dir.parent / "skills"
        )

        # Initialize agents
        self.learning_agent = LearningAgent(self.context_manager)
        self.performance_agent = PerformanceAgent(self.context_manager)
        self.research_agent = ResearchAgent(self.context_manager)

        self.agents = {
            "learning_agent": self.learning_agent,
            "performance_agent": self.performance_agent,
            "research_agent": self.research_agent,
        }

        # Initialize orchestrator LLM
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.model = model

        # Initialize recorder
        self.recorder = get_conversation_recorder()

        # Current state
        self.state: Optional[OrchestratorState] = None

    def _call_orchestrator_llm(
        self,
        messages: List[Dict],
        system: str = None,
    ) -> str:
        """Call the orchestrator's LLM."""
        if system is None:
            system = ORCHESTRATOR_SYSTEM_PROMPT

        # Add context from memory
        context = self.context_manager.get_context_for_agent(
            agent_id="orchestrator",
            query=messages[-1]["content"] if messages else "",
        )

        # Prepend RAG context
        full_messages = []
        if context["rag_memories"]:
            rag_text = "## Relevant Context\n"
            for mem in context["rag_memories"]:
                rag_text += f"- [{mem['category']}] {mem['content'][:200]}...\n"
            full_messages.append({"role": "user", "content": rag_text})
            full_messages.append({"role": "assistant", "content": "I've reviewed the context."})

        full_messages.extend(messages)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=full_messages,
        )

        return response.content[0].text

    def _parse_handoff(self, response: str) -> Optional[Dict]:
        """Parse a handoff instruction from orchestrator response."""
        if "HANDOFF TO:" not in response:
            return None

        try:
            # Extract agent name
            lines = response.split("\n")
            agent_line = [l for l in lines if "HANDOFF TO:" in l][0]
            agent = agent_line.split("HANDOFF TO:")[1].strip().lower()
            agent = agent.replace(" ", "_")

            # Normalize agent name
            agent_map = {
                "learning": "learning_agent",
                "learning_agent": "learning_agent",
                "performance": "performance_agent",
                "performance_agent": "performance_agent",
                "research": "research_agent",
                "research_agent": "research_agent",
            }
            agent = agent_map.get(agent, agent)

            # Extract task
            task_start = response.find("TASK:")
            context_start = response.find("CONTEXT:")
            task = ""
            if task_start > -1:
                task_end = context_start if context_start > task_start else len(response)
                task = response[task_start + 5:task_end].strip()

            return {
                "agent": agent,
                "task": task,
                "full_instruction": response,
            }

        except Exception as e:
            print(f"Failed to parse handoff: {e}")
            return None

    def _execute_handoff(self, handoff: Dict) -> str:
        """Execute a handoff to an agent."""
        agent_id = handoff["agent"]

        if agent_id not in self.agents:
            return f"Unknown agent: {agent_id}"

        agent = self.agents[agent_id]

        # Create handoff message
        handoff_msg = self.context_manager.handoff_to_agent(
            from_agent="orchestrator",
            to_agent=agent_id,
            instructions=handoff["task"],
            context_summary=self.context_manager.summarize_conversation(),
        )

        # Record handoff
        self.recorder.orchestrator_acts(
            f"Handoff to {agent_id}",
            {"task": handoff["task"][:100]},
        )

        # Have agent think about the task
        result = agent.think(handoff["full_instruction"])

        # Record result
        if agent_id == "learning_agent":
            self.recorder.learning_agent_returns(result[:500])
        elif agent_id == "performance_agent":
            self.recorder.performance_agent_returns(result[:500])
        elif agent_id == "research_agent":
            self.recorder.research_agent_returns(result[:500])

        # Clear handoff after processing
        self.context_manager.clear_agent_handoff(agent_id)

        return result

    def process_user_input(self, user_input: str) -> str:
        """
        Process a user input through the multi-agent system.

        Returns the final response.
        """
        # Start recording if not already
        if self.recorder.current is None:
            self.recorder.start_conversation(user_input[:50])

        # Record user input
        self.recorder.user_says(user_input)

        # Add to context
        user_msg = AgentMessage(
            source="user",
            message_type=MessageType.USER_INPUT.value,
            content=user_input,
            importance=0.8,
        )
        self.context_manager.add_message(user_msg, memory_category="task")

        # Phase 1: Orchestrator understands and plans
        messages = [{"role": "user", "content": user_input}]
        orchestrator_response = self._call_orchestrator_llm(messages)

        # Record orchestrator thinking
        self.recorder.orchestrator_thinks(orchestrator_response[:300])

        # Store orchestrator response
        orch_msg = AgentMessage(
            source="orchestrator",
            message_type=MessageType.PLAN.value,
            content=orchestrator_response,
            importance=0.7,
        )
        self.context_manager.add_message(orch_msg, memory_category="planning")

        # Check for handoff
        handoff = self._parse_handoff(orchestrator_response)

        if handoff:
            # Execute handoff
            agent_result = self._execute_handoff(handoff)

            # Phase 2: Orchestrator synthesizes result
            synthesis_messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": orchestrator_response},
                {"role": "user", "content": f"Agent {handoff['agent']} responded:\n\n{agent_result}"},
            ]

            final_response = self._call_orchestrator_llm(
                synthesis_messages,
                system=ORCHESTRATOR_SYSTEM_PROMPT + "\n\nNow synthesize the agent's response into a clear, helpful response for the user.",
            )

            # Check for additional handoffs (iterative)
            iteration = 0
            while iteration < 3:
                next_handoff = self._parse_handoff(final_response)
                if not next_handoff:
                    break

                agent_result = self._execute_handoff(next_handoff)

                synthesis_messages.append({"role": "assistant", "content": final_response})
                synthesis_messages.append({"role": "user", "content": f"Agent {next_handoff['agent']} responded:\n\n{agent_result}"})

                final_response = self._call_orchestrator_llm(
                    synthesis_messages,
                    system=ORCHESTRATOR_SYSTEM_PROMPT + "\n\nContinue coordinating agents or provide final response to user.",
                )

                iteration += 1

        else:
            final_response = orchestrator_response

        # Record final response
        self.recorder.orchestrator_says(final_response)

        # Store in context
        final_msg = AgentMessage(
            source="orchestrator",
            message_type=MessageType.RESPONSE.value,
            content=final_response,
            importance=0.6,
        )
        self.context_manager.add_message(final_msg, memory_category="response")

        return final_response

    def run_interactive(self):
        """Run interactive session."""
        print("=" * 60)
        print("Multi-Agent Robot Skill Learning System")
        print("=" * 60)
        print()
        print("I coordinate three specialized agents:")
        print("  Learning Agent     - Designs and trains skills")
        print("  Performance Agent  - Tests skills in simulation")
        print("  Research Agent     - Analyzes and explains policies")
        print()
        print("Tell me what you want the robot to learn!")
        print("Type 'quit' to exit, 'save' to save conversation.")
        print("-" * 60)
        print()

        self.recorder.start_conversation("Interactive Session")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "quit":
                    self._end_session()
                    break

                if user_input.lower() == "save":
                    json_path, md_path = self.recorder.end_conversation()
                    print(f"\nSaved to: {md_path}\n")
                    self.recorder.start_conversation("Continued Session")
                    continue

                if user_input.lower() == "history":
                    self._show_history()
                    continue

                print("\nOrchestrator: ", end="", flush=True)
                response = self.process_user_input(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                self._end_session()
                break

    def _end_session(self):
        """End the current session."""
        print("\n\nSaving conversation...")
        self.recorder.set_summary(self.context_manager.summarize_conversation())
        json_path, md_path = self.recorder.end_conversation()
        if md_path:
            print(f"Saved to: {md_path}")
        print("Goodbye!")

    def _show_history(self):
        """Show conversation history."""
        print("\nConversation History:\n")
        for msg in self.context_manager.global_history[-20:]:
            print(f"[{msg.source}] {msg.content[:80]}...")
        print()


def create_multi_agent_orchestrator(persist_dir: str = None) -> MultiAgentOrchestrator:
    """Create a multi-agent orchestrator instance."""
    if persist_dir:
        persist_dir = Path(persist_dir)
    return MultiAgentOrchestrator(persist_dir=persist_dir)
