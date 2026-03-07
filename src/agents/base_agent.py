"""
Base Agent Class

Each agent has:
- Its own LLM instance
- Its own context window
- Access to shared memory (RAG)
- Specialized system prompt
- Specific tools via MCP
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json

import anthropic
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context.context_manager import ContextManager
from src.context.message_types import AgentMessage, MessageType

# Load environment
load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    name: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    Each agent:
    - Has its own Claude instance
    - Maintains conversation history
    - Can call tools via MCP
    - Receives/sends context to other agents
    """

    def __init__(
        self,
        config: AgentConfig,
        context_manager: ContextManager,
    ):
        self.config = config
        self.context_manager = context_manager

        # Initialize Claude client
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        # Tool definitions (override in subclasses)
        self.tools: List[Dict] = []

        # Conversation tracking
        self.conversation_id: Optional[str] = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass

    def _build_messages(self, user_message: str, query: str = None) -> List[Dict]:
        """Build messages array for LLM call."""
        # Get context (RAG + recent)
        context = self.context_manager.get_context_for_agent(
            agent_id=self.config.agent_id,
            query=query or user_message,
            include_rag=True,
            include_recent=True,
        )

        messages = []

        # Add RAG context as system context
        if context["rag_memories"]:
            rag_context = "## Relevant Context from Memory\n\n"
            for mem in context["rag_memories"]:
                rag_context += f"**[{mem['category']}]** {mem['content'][:300]}...\n\n"
            messages.append({
                "role": "user",
                "content": f"<context>\n{rag_context}</context>",
            })
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the relevant context. How can I help?",
            })

        # Add recent conversation
        messages.extend(context["recent_messages"])

        # Add current message
        messages.append({
            "role": "user",
            "content": user_message,
        })

        return messages

    def think(self, prompt: str, **kwargs) -> str:
        """
        Have the agent think about something.

        Returns the agent's response.
        """
        # Build system prompt with current context
        system = self.system_prompt
        system_context = self.context_manager.build_system_context(self.config.agent_id)
        if system_context:
            system = f"{system}\n\n{system_context}"

        # Build messages
        messages = self._build_messages(prompt)

        # Call Claude
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=messages,
            tools=self.tools if self.tools else None,
        )

        # Extract text response
        text_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text_content += block.text

        # Store in context
        thought = AgentMessage(
            source=self.config.agent_id,
            message_type=MessageType.THOUGHT.value,
            content=text_content,
            importance=0.6,
        )
        self.context_manager.add_message(thought, memory_category="reasoning")

        return text_content

    def act(self, instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Have the agent take an action (call tools).

        Returns action results.
        """
        # Build system prompt
        system = self.system_prompt
        system_context = self.context_manager.build_system_context(self.config.agent_id)
        if system_context:
            system = f"{system}\n\n{system_context}"

        # Build messages
        messages = self._build_messages(instruction)

        # Call Claude with tools
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=0.3,  # Lower for actions
            system=system,
            messages=messages,
            tools=self.tools if self.tools else None,
        )

        results = {
            "text": "",
            "tool_calls": [],
            "stop_reason": response.stop_reason,
        }

        for block in response.content:
            if hasattr(block, 'text'):
                results["text"] += block.text
            elif block.type == "tool_use":
                results["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return results

    def receive_handoff(self, handoff: AgentMessage):
        """Process a handoff from another agent."""
        self.context_manager.agent_contexts[self.config.agent_id].last_handoff = handoff

    def send_handoff(
        self,
        to_agent: str,
        instructions: str,
        context_summary: str,
    ) -> AgentMessage:
        """Send a handoff to another agent."""
        return self.context_manager.handoff_to_agent(
            from_agent=self.config.agent_id,
            to_agent=to_agent,
            instructions=instructions,
            context_summary=context_summary,
        )


# =============================================================================
# LEARNING AGENT
# =============================================================================

LEARNING_AGENT_PROMPT = """You are the **Learning Agent** for a humanoid robot skill learning system.

## Your Role
You are an expert in reinforcement learning and robot skill acquisition. Your job is to:
1. Break down complex motor skills into learnable sub-skills
2. Design reward functions that guide learning
3. Set up and monitor RL training experiments
4. Apply transfer learning from related skills
5. Troubleshoot training issues

## Robot Specification
- **Robot**: Unitree G1 Humanoid with dexterous hands
- **DOF**: 44 joints (legs, torso, arms, fingers)
- **Actuators**: 43 controllable motors
- **Observation**: Joint positions, velocities, IMU, contact sensors

## Your Expertise

### Skill Decomposition
When asked to train a complex skill like "walk and jump":
1. Identify prerequisite skills (balance, weight shift)
2. Order skills by dependency (balance → walk → jump)
3. Define clear success criteria for each
4. Suggest transfer learning opportunities

### Reward Function Design
For each skill, you design reward components:
- **height_reward**: Maintain/achieve target height
- **upright_reward**: Keep torso vertical
- **velocity_forward**: Move in desired direction
- **energy_efficiency**: Minimize control effort
- **stability**: Penalize jerky movements
- **foot_contact**: Proper gait patterns

### Training Configuration
You set hyperparameters based on skill complexity:
- **Simple skills** (balance): 200K steps, lr=3e-4
- **Medium skills** (walk): 500K-1M steps, lr=1e-4
- **Complex skills** (backflip): 2M+ steps, curriculum learning

### Transfer Learning Strategy
- Identify source skills with similar reward components
- Transfer policy network weights (not value function)
- Fine-tune with lower learning rate (1e-5)
- Monitor for catastrophic forgetting

## Output Format
When creating a skill, provide:
```yaml
skill_id: walk_forward
name: Walk Forward
description: Walk forward maintaining balance

reward_function:
  - component: velocity_forward
    weight: 1.0
  - component: upright_reward
    weight: 0.5
  - component: energy_efficiency
    weight: 0.1

training_config:
  algorithm: PPO
  total_timesteps: 500000
  learning_rate: 3e-4
  batch_size: 64

transfer_from: balance_stand  # If applicable

success_criteria:
  - forward_velocity > 0.5 m/s
  - episode_length > 500 steps

curriculum:  # If needed
  - stage: 1
    max_velocity: 0.2
  - stage: 2
    max_velocity: 0.5
```

## Important Guidelines
1. Always start with foundational skills before complex ones
2. Design rewards that are dense (frequent feedback) and shaped
3. Use curriculum learning for difficult skills
4. Monitor for reward hacking and training instabilities
5. Log everything for reproducibility
"""


class LearningAgent(BaseAgent):
    """
    Agent responsible for training robot skills.

    Specializes in:
    - Skill decomposition
    - Reward function design
    - RL experiment setup
    - Transfer learning
    """

    def __init__(self, context_manager: ContextManager):
        config = AgentConfig(
            agent_id="learning_agent",
            name="Learning Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
        )
        super().__init__(config, context_manager)

        # Define tools this agent can use
        self.tools = [
            {
                "name": "create_skill",
                "description": "Create a new skill definition for training",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "reward_components": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "training_config": {"type": "object"},
                        "transfer_from": {"type": "string"},
                    },
                    "required": ["skill_id", "name", "reward_components"],
                },
            },
            {
                "name": "start_training",
                "description": "Start training a skill",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "config_overrides": {"type": "object"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "check_training_progress",
                "description": "Check the progress of ongoing training",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "analyze_reward_curve",
                "description": "Analyze the reward curve for issues",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                    },
                    "required": ["skill_id"],
                },
            },
        ]

    @property
    def system_prompt(self) -> str:
        return LEARNING_AGENT_PROMPT


# =============================================================================
# PERFORMANCE AGENT
# =============================================================================

PERFORMANCE_AGENT_PROMPT = """You are the **Performance Agent** for a humanoid robot simulation system.

## Your Role
You control and observe the robot in the MuJoCo physics simulation. Your job is to:
1. Execute trained skills on the robot
2. Run simulations and collect performance data
3. Test skill robustness under different conditions
4. Report observations and metrics back to other agents

## Robot Specification
- **Robot**: Unitree G1 Humanoid with dexterous hands
- **Simulation**: MuJoCo physics engine
- **Control**: Position/torque control at 50Hz
- **Sensors**: Joint encoders, IMU, contact sensors

## Your Capabilities

### Simulation Control
- Reset robot to initial pose
- Step simulation forward
- Apply actions from trained policies
- Modify simulation parameters (friction, mass, etc.)

### Skill Execution
- Load trained skill policies
- Execute skills with deterministic or stochastic actions
- Chain multiple skills in sequence
- Handle skill transitions

### Data Collection
For each execution, you track:
- Episode length and total reward
- Center of mass trajectory
- Joint positions and velocities
- Contact forces
- Energy consumption

### Testing Scenarios
You can test skills under:
- Nominal conditions
- External perturbations (pushes)
- Terrain variations
- Sensor noise

## Output Format
When reporting execution results:
```yaml
execution_report:
  skill_id: walk_forward
  episodes: 10

  metrics:
    mean_reward: 245.3
    std_reward: 12.1
    mean_episode_length: 847
    success_rate: 0.9

  observations:
    - Robot maintained balance throughout
    - Slight drift to the left
    - Smooth gait pattern achieved

  failure_modes:
    - 1 episode: Lost balance at step 234

  recommendations:
    - Consider adding drift correction
    - Skill ready for transfer learning
```

## Important Guidelines
1. Always reset simulation before new tests
2. Run multiple episodes for statistical significance
3. Report both successes and failures
4. Note any unexpected behaviors
5. Collect data for analysis by Research Agent
"""


class PerformanceAgent(BaseAgent):
    """
    Agent responsible for running robot simulations.

    Specializes in:
    - Skill execution
    - Performance testing
    - Data collection
    - Robustness testing
    """

    def __init__(self, context_manager: ContextManager):
        config = AgentConfig(
            agent_id="performance_agent",
            name="Performance Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3,
        )
        super().__init__(config, context_manager)

        self.tools = [
            {
                "name": "reset_simulation",
                "description": "Reset the robot simulation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer"},
                    },
                },
            },
            {
                "name": "execute_skill",
                "description": "Execute a trained skill",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "n_steps": {"type": "integer"},
                        "deterministic": {"type": "boolean"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "run_evaluation",
                "description": "Run full evaluation of a skill",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "n_episodes": {"type": "integer"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "get_robot_state",
                "description": "Get current robot state",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "apply_perturbation",
                "description": "Apply external force to test robustness",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "force": {"type": "array"},
                        "body": {"type": "string"},
                    },
                },
            },
        ]

    @property
    def system_prompt(self) -> str:
        return PERFORMANCE_AGENT_PROMPT


# =============================================================================
# RESEARCH AGENT
# =============================================================================

RESEARCH_AGENT_PROMPT = """You are the **Research Agent** for a robotics explainable AI (XAI) system.

## Your Role
You are an expert in interpretable machine learning and robot behavior analysis. Your job is to:
1. Explain why learned policies make specific decisions
2. Analyze skill performance and identify issues
3. Compare different training approaches
4. Generate insights for improving robot learning
5. Create reports for human understanding

## Your Expertise

### Explainability Methods (XAI)
You apply multiple techniques:

**Saliency Analysis**
- Compute input gradients to identify important observations
- Show which joint positions/velocities influence decisions
- Visualize attention over time

**Feature Importance**
- Aggregate importance across episodes
- Identify which sensors the policy relies on
- Detect unused or redundant inputs

**Action Distribution Analysis**
- Analyze policy entropy (exploration vs. exploitation)
- Identify deterministic vs. uncertain decisions
- Map observation regions to action patterns

**Counterfactual Analysis**
- "What if joint X was different?"
- Identify critical state variables
- Find decision boundaries

### Performance Analysis
You analyze training and execution data:

**Learning Curves**
- Detect plateaus and instabilities
- Identify reward hacking
- Compare to baselines

**Failure Mode Analysis**
- Categorize failure types
- Find common preconditions
- Suggest mitigations

**Skill Comparison**
- Compare policies trained differently
- Analyze transfer learning effects
- Benchmark against prior work

### Insight Generation
You synthesize findings into actionable insights:
- Why is the robot falling? → "Policy over-relies on velocity, ignores contacts"
- Why is training slow? → "Sparse reward, consider shaping"
- How to improve? → "Add curriculum, transfer from balance skill"

## Output Format
When providing analysis:
```yaml
analysis_report:
  skill_id: walk_forward
  analysis_type: failure_mode

  findings:
    - finding: Robot falls when turning
      evidence: 85% of failures occur during direction change
      explanation: Policy not trained on turning scenarios

    - finding: Right leg weaker than left
      evidence: Right hip torque 20% lower on average
      explanation: Asymmetric reward or initialization

  xai_insights:
    top_features:
      - com_velocity_x (importance: 0.45)
      - left_ankle_angle (importance: 0.23)
      - torso_pitch (importance: 0.18)

    policy_behavior:
      - High certainty in forward walking
      - Uncertain near obstacles

  recommendations:
    - priority: high
      action: Add turning scenarios to training
      expected_impact: Reduce fall rate by 50%

    - priority: medium
      action: Initialize with symmetric pose
      expected_impact: Balance left/right performance
```

## Important Guidelines
1. Always provide evidence for claims
2. Make explanations accessible to non-experts
3. Prioritize actionable recommendations
4. Connect XAI findings to training improvements
5. Track insights over time for meta-learning
"""


class ResearchAgent(BaseAgent):
    """
    Agent responsible for analysis and explainability.

    Specializes in:
    - XAI / Interpretability
    - Performance analysis
    - Failure mode detection
    - Insight generation
    """

    def __init__(self, context_manager: ContextManager):
        config = AgentConfig(
            agent_id="research_agent",
            name="Research Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.6,
        )
        super().__init__(config, context_manager)

        self.tools = [
            {
                "name": "analyze_skill",
                "description": "Run comprehensive skill analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "analysis_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "explain_decision",
                "description": "Explain a specific policy decision using XAI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "observation": {"type": "array"},
                        "method": {"type": "string"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "compare_skills",
                "description": "Compare two skills",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id_1": {"type": "string"},
                        "skill_id_2": {"type": "string"},
                    },
                    "required": ["skill_id_1", "skill_id_2"],
                },
            },
            {
                "name": "analyze_failures",
                "description": "Analyze failure modes",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "n_episodes": {"type": "integer"},
                    },
                    "required": ["skill_id"],
                },
            },
            {
                "name": "generate_report",
                "description": "Generate comprehensive skill report",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "include_xai": {"type": "boolean"},
                    },
                    "required": ["skill_id"],
                },
            },
        ]

    @property
    def system_prompt(self) -> str:
        return RESEARCH_AGENT_PROMPT
