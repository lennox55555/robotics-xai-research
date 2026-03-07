"""
Orchestrator Agent

The main brain that coordinates all sub-agents:
- Learning Agent: Trains new skills
- Performance Agent: Executes skills in simulation
- Research Agent: Analyzes and explains policies

Uses LangGraph for agent orchestration and MCP for tool communication.
"""

import json
import operator
from typing import Annotated, List, Dict, Any, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.skill_learning.skill import Skill, SkillLibrary, TaskDecomposition
from src.orchestrator.llm_orchestrator import LLMOrchestrator, create_g1_robot_spec


# State definition for the orchestrator
class OrchestratorState(TypedDict):
    """State maintained by the orchestrator."""
    messages: Annotated[List, operator.add]
    task_prompt: str
    task_decomposition: Optional[Dict]
    current_skill: Optional[str]
    skills_to_train: List[str]
    skills_trained: List[str]
    agent_responses: Dict[str, Any]
    next_action: str
    iteration: int


# System prompt for the orchestrator
ORCHESTRATOR_SYSTEM = """You are the Orchestrator Agent for a humanoid robot learning system.

## Your Role
You coordinate three specialized agents to help a robot learn new skills:

1. **Learning Agent**: Trains new RL policies for skills
   - Creates skill definitions
   - Runs training jobs
   - Handles transfer learning

2. **Performance Agent**: Executes skills in simulation
   - Runs the robot simulation
   - Tests trained skills
   - Collects performance data

3. **Research Agent**: Analyzes and improves
   - Explains policy decisions (XAI)
   - Compares skill performance
   - Suggests improvements

## Your Workflow

When given a task like "teach the robot to walk and jump":

1. **Decompose**: Break the task into learnable skills
2. **Plan**: Determine training order and dependencies
3. **Train**: Use Learning Agent to train each skill
4. **Test**: Use Performance Agent to verify skills work
5. **Analyze**: Use Research Agent to understand and improve
6. **Compose**: Combine skills into final behavior

## Available Tools

You have access to tools from all three agents. Use them strategically:

- Start with `list_skills` to see what's already trained
- Use `create_skill` to define new skills
- Use `train_skill` to start training
- Use `execute_skill` to test in simulation
- Use `analyze_skill_performance` to evaluate
- Use `explain_policy_decision` for XAI insights

## Important Guidelines

1. Always check existing skills before training new ones
2. Train prerequisite skills first (e.g., balance before walk)
3. Use transfer learning when skills are similar
4. Test skills after training before moving on
5. Analyze failures to suggest improvements

Respond conversationally but be action-oriented. After analysis, always take the next step.
"""


class OrchestratorAgent:
    """
    Main orchestrator that coordinates skill learning.

    Uses LangGraph for state management and routing between agents.
    """

    def __init__(
        self,
        skills_dir: Path,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.skills_dir = Path(skills_dir)
        self.model = model

        # Initialize components
        self.skill_library = SkillLibrary(skills_dir)
        self.robot_spec = create_g1_robot_spec()
        self.llm_orchestrator = LLMOrchestrator(
            self.robot_spec,
            self.skill_library,
            model=model,
        )

        # Initialize LLM
        self.llm = ChatAnthropic(model=model)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        # Define the graph
        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("analyze_task", self._analyze_task)
        workflow.add_node("plan_training", self._plan_training)
        workflow.add_node("train_skill", self._train_skill)
        workflow.add_node("test_skill", self._test_skill)
        workflow.add_node("analyze_results", self._analyze_results)
        workflow.add_node("respond", self._respond)

        # Set entry point
        workflow.set_entry_point("analyze_task")

        # Add edges
        workflow.add_conditional_edges(
            "analyze_task",
            self._route_after_analysis,
            {
                "plan": "plan_training",
                "respond": "respond",
            }
        )

        workflow.add_edge("plan_training", "train_skill")

        workflow.add_conditional_edges(
            "train_skill",
            self._route_after_training,
            {
                "test": "test_skill",
                "train_more": "train_skill",
                "respond": "respond",
            }
        )

        workflow.add_edge("test_skill", "analyze_results")

        workflow.add_conditional_edges(
            "analyze_results",
            self._route_after_analysis_results,
            {
                "train_more": "train_skill",
                "respond": "respond",
            }
        )

        workflow.add_edge("respond", END)

        return workflow.compile()

    def _analyze_task(self, state: OrchestratorState) -> Dict:
        """Analyze the incoming task and decompose if needed."""
        task_prompt = state["task_prompt"]

        # Use LLM orchestrator to decompose task
        try:
            decomposition = self.llm_orchestrator.decompose_task(task_prompt)

            return {
                "task_decomposition": decomposition.to_dict(),
                "skills_to_train": decomposition.execution_order,
                "messages": [AIMessage(content=f"I've analyzed your task and identified {len(decomposition.skills)} skills to train: {', '.join(decomposition.execution_order)}")],
                "next_action": "plan",
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"I had trouble analyzing that task: {e}")],
                "next_action": "respond",
            }

    def _plan_training(self, state: OrchestratorState) -> Dict:
        """Plan the training sequence."""
        skills_to_train = state["skills_to_train"]
        skills_trained = state.get("skills_trained", [])

        # Find next skill to train
        remaining = [s for s in skills_to_train if s not in skills_trained]

        if not remaining:
            return {
                "messages": [AIMessage(content="All skills have been trained!")],
                "current_skill": None,
                "next_action": "respond",
            }

        next_skill = remaining[0]
        return {
            "current_skill": next_skill,
            "messages": [AIMessage(content=f"Starting training for skill: {next_skill}")],
        }

    def _train_skill(self, state: OrchestratorState) -> Dict:
        """Train the current skill."""
        skill_id = state["current_skill"]

        if not skill_id:
            return {
                "next_action": "respond",
                "messages": [AIMessage(content="No skill selected for training.")],
            }

        # Get skill from library
        skill = self.skill_library.get_skill(skill_id)

        if skill is None:
            return {
                "next_action": "respond",
                "messages": [AIMessage(content=f"Skill '{skill_id}' not found.")],
            }

        # In real implementation, this would call the Learning Agent MCP
        # For now, simulate training
        return {
            "messages": [AIMessage(content=f"Training '{skill_id}'... (This would call the Learning Agent)")],
            "next_action": "test",
        }

    def _test_skill(self, state: OrchestratorState) -> Dict:
        """Test the trained skill."""
        skill_id = state["current_skill"]

        # In real implementation, this would call the Performance Agent MCP
        return {
            "messages": [AIMessage(content=f"Testing '{skill_id}' in simulation... (This would call the Performance Agent)")],
            "agent_responses": {
                "performance_test": {
                    "skill_id": skill_id,
                    "mean_reward": 100.0,
                    "episodes": 10,
                }
            },
        }

    def _analyze_results(self, state: OrchestratorState) -> Dict:
        """Analyze training/test results."""
        skill_id = state["current_skill"]
        skills_trained = state.get("skills_trained", [])

        # Mark skill as trained
        skills_trained = skills_trained + [skill_id]

        # Check if more skills to train
        skills_to_train = state["skills_to_train"]
        remaining = [s for s in skills_to_train if s not in skills_trained]

        if remaining:
            next_action = "train_more"
            message = f"Skill '{skill_id}' trained successfully! Moving to next skill: {remaining[0]}"
        else:
            next_action = "respond"
            message = f"All skills trained! Task complete."

        return {
            "skills_trained": skills_trained,
            "current_skill": remaining[0] if remaining else None,
            "messages": [AIMessage(content=message)],
            "next_action": next_action,
        }

    def _respond(self, state: OrchestratorState) -> Dict:
        """Generate final response."""
        # Compile summary
        skills_trained = state.get("skills_trained", [])
        task_decomposition = state.get("task_decomposition", {})

        summary = f"""
## Task Summary

**Original Task**: {state.get('task_prompt', 'N/A')}

**Skills Trained**: {', '.join(skills_trained) if skills_trained else 'None'}

**Status**: {'Complete' if len(skills_trained) == len(state.get('skills_to_train', [])) else 'In Progress'}
"""
        return {
            "messages": [AIMessage(content=summary)],
        }

    def _route_after_analysis(self, state: OrchestratorState) -> str:
        """Route after task analysis."""
        return state.get("next_action", "respond")

    def _route_after_training(self, state: OrchestratorState) -> str:
        """Route after training a skill."""
        return state.get("next_action", "test")

    def _route_after_analysis_results(self, state: OrchestratorState) -> str:
        """Route after analyzing results."""
        return state.get("next_action", "respond")

    def run(self, task_prompt: str) -> Dict[str, Any]:
        """
        Run the orchestrator on a task.

        Args:
            task_prompt: Natural language description of the task
                        e.g., "Teach the robot to walk forward and then jump"

        Returns:
            Final state with results
        """
        initial_state = {
            "messages": [HumanMessage(content=task_prompt)],
            "task_prompt": task_prompt,
            "task_decomposition": None,
            "current_skill": None,
            "skills_to_train": [],
            "skills_trained": [],
            "agent_responses": {},
            "next_action": "",
            "iteration": 0,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return final_state

    def chat(self, message: str) -> str:
        """
        Chat interface for interactive use.

        Args:
            message: User message

        Returns:
            Agent response
        """
        result = self.run(message)

        # Extract final messages
        messages = result.get("messages", [])
        ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]

        return "\n\n".join(ai_messages)


def create_orchestrator(skills_dir: str = None) -> OrchestratorAgent:
    """Create an orchestrator agent instance."""
    if skills_dir is None:
        skills_dir = Path(__file__).parent.parent.parent.parent / "skills"

    return OrchestratorAgent(Path(skills_dir))
