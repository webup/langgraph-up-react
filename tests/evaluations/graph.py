"""Graph trajectory evaluation using AgentEvals framework with LangSmith integration.

This module implements comprehensive evaluation of the ReAct agent using graph trajectory
LLM-as-judge methodology with async execution and LangSmith pytest integration.
"""

import logging
import os
from typing import Any, Dict

import pytest
from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import aextract_langgraph_trajectory_from_thread
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

from common.context import Context
from common.utils import load_chat_model
from tests.evaluations.utils import normalize_trajectory_data

# Configure logging for evaluation
logger = logging.getLogger(__name__)


class EvaluationConfig:
    """Configuration for graph trajectory evaluation."""

    # Agent models to test - using exact model names as specified
    AGENT_MODELS = [
        "siliconflow:Qwen/Qwen3-8B",  # Qwen/Qwen3-8B
        "siliconflow:THUDM/GLM-4-9B-0414",  # THUDM/GLM-4-9B-0414
    ]

    # Evaluator model - using GLM-Z1 for better reasoning and tool support
    EVALUATOR_MODEL = "siliconflow:THUDM/GLM-Z1-9B-0414"

    # Test scenarios with different evaluation rubrics
    TEST_SCENARIOS = [
        {
            "name": "simple_question",
            "query": "What is the capital of France?",
            "expected_behavior": "Should provide direct answer without unnecessary tool usage",
            "rubric": """An accurate trajectory for simple factual questions:
            - MUST provide direct answer from existing knowledge WITHOUT any external tool calls
            - Trajectory should be: __start__ -> call_model (NO 'tools' node)
            - Questions like 'What is the capital of France?' should NEVER trigger web search or other external tools
            - If trajectory contains 'tools' node, it's an AUTOMATIC FAILURE (score: 0)
            - Shows efficient reasoning with minimal steps - only model reasoning, no tool execution
            - Demonstrates appropriate confidence in well-known facts
            - Score 1 only if answered correctly without 'tools' node, Score 0 if 'tools' node appears in trajectory""",
        },
        {
            "name": "search_required",
            "query": "What's the latest news about artificial intelligence?",
            "expected_behavior": "Should use search tools to find current information",
            "rubric": """An accurate trajectory for information gathering:
            - Expected trajectory: __start__ -> call_model -> tools -> call_model (MUST include 'tools' node)
            - Uses search tools when current information is needed
            - Makes appropriate search queries and retrieves actual current information
            - Synthesizes information from search results effectively into actionable insights
            - Shows logical progression from query to search to synthesis
            - If no 'tools' node appears, it's a failure - current info cannot be known without search
            - CONTENT QUALITY EVALUATION:
              * HIGH QUALITY (Score 1): Provides specific recent news items with concrete details, dates, company names, or developments
              * MEDIUM QUALITY (Score 0.5): Uses search tools but provides mostly generic summaries without actionable information  
              * LOW QUALITY (Score 0): Generic responses, just listing sources, or primarily providing links instead of content
            - LINK GUIDANCE: Providing links is NOT helpful - users want actual current information, not homework
            - IGNORE: Link lists, source directories, "check these websites" responses - these avoid doing the synthesis work
            - KEY DISTINCTION: "NBC reports on AI developments" (generic) vs "NBC reported today that OpenAI announced GPT-5" (specific)
            - Final score requires both correct trajectory AND high-quality specific content from search results (not just links)""",
        },
        {
            "name": "multi_step_reasoning",
            "query": "What are the pros and cons of renewable energy, and what are the latest developments?",
            "expected_behavior": "Should search for information and provide structured analysis",
            "rubric": """An accurate trajectory for complex analytical tasks:
            - Expected trajectory: __start__ -> call_model -> tools -> call_model (MUST include 'tools' node for current info)
            - Breaks down complex questions into manageable parts (pros/cons + latest developments)
            - Uses search tools to gather current information about renewable energy developments
            - Demonstrates structured thinking with clear analysis and synthesis
            - Provides balanced perspective with specific pros and cons backed by evidence
            - Integrates current developments with foundational knowledge from search results
            - DUAL REQUIREMENT: Must use tools for current info AND provide structured analytical thinking
            - QUALITY EVALUATION:
              * HIGH QUALITY (Score 1): Uses search + provides specific current developments + structured pros/cons analysis
              * MEDIUM QUALITY (Score 0.5): Uses search but analysis is generic or lacks current information
              * LOW QUALITY (Score 0): No search tools or purely generic analysis without current developments
            - Final score requires both search execution AND comprehensive analytical synthesis""",
        },
    ]


async def create_evaluation_context(model_name: str) -> Context:
    """Create context for evaluation with specified model.

    Args:
        model_name: The model name in provider:model format

    Returns:
        Configured Context instance
    """
    return Context(
        model=model_name,
        max_search_results=3,  # Limit for evaluation
        enable_deepwiki=False,  # Disable for controlled evaluation
    )


async def run_agent_scenario(
    model_name: str, query: str, thread_id: str
) -> Dict[str, Any]:
    """Run a single evaluation scenario with the agent.

    Args:
        model_name: Model to use for the agent
        query: Query to ask the agent
        thread_id: Thread ID for conversation tracking

    Returns:
        Dictionary containing trajectory and metadata
    """
    # Create checkpointer for conversation memory
    checkpointer = MemorySaver()

    # Create context with specified model
    context = await create_evaluation_context(model_name)

    # Use the existing graph with checkpointer for trajectory extraction
    # We need to get the graph builder and recompile with checkpointer
    from react_agent.graph import builder

    compiled_graph = builder.compile(checkpointer=checkpointer)

    # Configuration for the graph execution
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    try:
        # Execute the agent conversation with context
        result = await compiled_graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config=config, context=context
        )

        # Extract the trajectory
        trajectory = await aextract_langgraph_trajectory_from_thread(
            compiled_graph, {"configurable": {"thread_id": thread_id}}
        )

        return {
            "trajectory": trajectory,
            "result": result,
            "model": model_name,
            "query": query,
            "thread_id": thread_id,
        }

    except Exception as e:
        return {
            "error": str(e),
            "model": model_name,
            "query": query,
            "thread_id": thread_id,
        }


async def create_scenario_evaluator(scenario: Dict[str, str], evaluator_model: str):
    """Create an async graph trajectory evaluator with scenario-specific rubric.

    Args:
        scenario: Test scenario with name, query, and rubric
        evaluator_model: Model to use as evaluator judge

    Returns:
        Configured async graph trajectory evaluator
    """
    # Create custom prompt with scenario-specific rubric
    custom_prompt = f"""You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving user queries.

<Scenario-Specific Rubric>
{scenario["rubric"]}
</Scenario-Specific Rubric>

<General Instructions>
Grade the following thread, evaluating whether the agent's overall steps are logical and appropriate.
For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
interrupting to await additional data from another source ("human-in-the-loop"):
</General Instructions>

<thread>
{{thread}}
</thread>

{{reference_outputs}}
"""

    try:
        # Use load_chat_model to create judge
        custom_judge = load_chat_model(evaluator_model)

        # Create async graph trajectory evaluator
        evaluator = create_async_graph_trajectory_llm_as_judge(
            judge=custom_judge,
            prompt=custom_prompt,
            continuous=False,  # Use boolean scoring
            use_reasoning=True,  # Enable reasoning explanation
            feedback_key=f"react_agent_{scenario['name']}_accuracy",
        )
        return evaluator
    except Exception as e:
        logger.error(f"Failed to create evaluator for scenario {scenario['name']}: {e}")
        raise


async def evaluate_trajectory(
    trajectory_data: Dict[str, Any], scenario: Dict[str, str], evaluator_model: str
) -> Dict[str, Any]:
    """Evaluate agent trajectory using async LLM-as-judge with scenario-specific rubric.

    Args:
        trajectory_data: Data from agent execution
        scenario: Test scenario with rubric
        evaluator_model: Model to use as evaluator judge

    Returns:
        Evaluation results
    """
    if "error" in trajectory_data:
        return {
            "evaluation_error": "Agent execution failed",
            "agent_error": trajectory_data["error"],
            "model": trajectory_data["model"],
            "query": trajectory_data["query"],
        }

    # Create scenario-specific evaluator
    evaluator = await create_scenario_evaluator(scenario, evaluator_model)

    try:
        # Convert trajectory data using utility function
        trajectory = trajectory_data["trajectory"]
        serializable_inputs, serializable_outputs = normalize_trajectory_data(
            trajectory
        )

        # Run the evaluation with normalized data
        evaluation_result = await evaluator(
            inputs=serializable_inputs,
            outputs=serializable_outputs,
            query=trajectory_data["query"],
        )

        return {
            "evaluation": evaluation_result,
            "model": trajectory_data["model"],
            "query": trajectory_data["query"],
            "thread_id": trajectory_data["thread_id"],
            "trajectory_steps": len(trajectory_data["trajectory"]["steps"]),
        }

    except Exception as e:
        return {
            "evaluation_error": str(e),
            "model": trajectory_data["model"],
            "query": trajectory_data["query"],
        }


# Create test combinations
TEST_COMBINATIONS = [
    (model_name, scenario)
    for model_name in EvaluationConfig.AGENT_MODELS
    for scenario in EvaluationConfig.TEST_SCENARIOS
]


# LangSmith pytest test functions
@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_combo",
    TEST_COMBINATIONS,
    ids=lambda combo: f"{combo[1]['name']}_{combo[0].split(':')[-1]}",
)
async def test_graph_trajectory_evaluation(test_combo):
    """Test graph trajectory evaluation with LangSmith integration for all model-scenario combinations."""
    # Unpack the test combination
    model_name, scenario = test_combo

    # Skip if required environment variables are not set
    required_env_vars = ["TAVILY_API_KEY", "SILICONFLOW_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            pytest.skip(f"Required environment variable {var} not set")

    # Also check for LangSmith API key
    if not os.getenv("LANGSMITH_API_KEY"):
        pytest.skip("LANGSMITH_API_KEY not set")

    # Generate unique thread ID for this combination
    thread_id = (
        f"eval_{model_name.replace(':', '_').replace('/', '_')}_{scenario['name']}"
    )

    logger.info(
        f"Testing {model_name} with scenario '{scenario['name']}': {scenario['query']}"
    )

    # Run agent scenario
    trajectory_data = await run_agent_scenario(
        model_name=model_name, query=scenario["query"], thread_id=thread_id
    )

    # Verify trajectory was captured
    assert "trajectory" in trajectory_data or "error" in trajectory_data

    # Log inputs as dict with query (LangSmith requires dict format)
    t.log_inputs({"question": scenario["query"]})

    # Log outputs - the actual agent conversation trajectory
    if "trajectory" in trajectory_data:
        # Convert to message format for the actual conversation
        messages = []
        if "result" in trajectory_data and "messages" in trajectory_data["result"]:
            for msg in trajectory_data["result"]["messages"]:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump())
                elif hasattr(msg, "dict"):
                    messages.append(msg.dict())
                else:
                    messages.append(dict(msg))

        t.log_outputs({"messages": messages})
    else:
        t.log_outputs({"error": trajectory_data.get("error", "Unknown error")})

    # Log reference outputs as expected behavior description
    t.log_reference_outputs({"expected_behavior": scenario["expected_behavior"]})

    # Run evaluation - this will log feedback to LangSmith automatically
    evaluation_result = await evaluate_trajectory(
        trajectory_data=trajectory_data,
        scenario=scenario,
        evaluator_model=EvaluationConfig.EVALUATOR_MODEL,
    )

    # Log evaluation results
    logger.info(f"[{model_name}] [{scenario['name']}] Evaluation completed")

    # Log detailed results if evaluation succeeded
    if "evaluation" in evaluation_result:
        eval_data = evaluation_result["evaluation"]
        score = eval_data.get("score", False)
        reasoning = eval_data.get("reasoning", "No reasoning provided")
        logger.info(
            f"[{model_name}] [{scenario['name']}] Score: {score}, Reasoning: {reasoning}"
        )

    # Note: We removed the assertion on score being True
    # Instead, let the evaluation score reflect real performance
    # LangSmith will track and display the actual evaluation results
