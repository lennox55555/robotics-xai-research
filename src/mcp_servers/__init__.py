"""
MCP (Model Context Protocol) Servers

Each agent exposes its capabilities through an MCP server:

1. Learning Agent (learning_server.py)
   - create_skill: Define new skills
   - train_skill: Train skills with RL
   - evaluate_skill: Evaluate trained skills
   - get_training_status: Check training progress

2. Performance Agent (performance_server.py)
   - reset_simulation: Reset robot state
   - step_simulation: Step the physics
   - execute_skill: Run a trained skill
   - get_robot_state: Get sensor data

3. Research Agent (research_server.py)
   - analyze_skill_performance: Detailed analysis
   - explain_policy_decision: XAI explanations
   - compare_skills: Compare two skills
   - suggest_improvements: Optimization hints

To run a server:
    python -m src.mcp_servers.learning_server
    python -m src.mcp_servers.performance_server
    python -m src.mcp_servers.research_server
"""
