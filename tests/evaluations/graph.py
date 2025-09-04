"""Graph trajectory evaluation using AgentEvals framework.

This module implements comprehensive evaluation of the ReAct agent using graph trajectory
LLM-as-judge methodology with async execution for performance.
"""

import logging
import os
from typing import Any, Dict

import pytest
from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import aextract_langgraph_trajectory_from_thread
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

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

    # Test scenarios
    TEST_SCENARIOS = [
        {
            "name": "simple_question",
            "query": "What is the capital of France?",
            "expected_behavior": "Should provide direct answer without unnecessary tool usage",
        },
        {
            "name": "search_required",
            "query": "What's the latest news about artificial intelligence?",
            "expected_behavior": "Should use search tools to find current information",
        },
        {
            "name": "multi_step_reasoning",
            "query": "What are the pros and cons of renewable energy, and what are the latest developments?",
            "expected_behavior": "Should search for information and provide structured analysis",
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


async def evaluate_trajectory(
    trajectory_data: Dict[str, Any], evaluator_model: str
) -> Dict[str, Any]:
    """Evaluate agent trajectory using LLM-as-judge.

    Args:
        trajectory_data: Data from agent execution
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

    # Custom prompt with JSON guardrails and boolean scoring
    CUSTOM_PROMPT = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

<Instructions>
  Grade the following thread, evaluating whether the agent's overall steps are logical and relatively efficient.
  For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
  interrupting to await additional data from another source ("human-in-the-loop"):
</Instructions>

You MUST respond with PURE JSON only.

CRITICAL: Your response must be EXACTLY in this JSON format with no other text:
{{"score": true, "reasoning": "Brief explanation here"}}

STRICT OUTPUT REQUIREMENTS:
- Return ONLY valid JSON: {{"score": <boolean>, "reasoning": "<text>"}}
- NO markdown, NO headers, NO extra text
- Keep reasoning under 50 words
- Score must be boolean: true or false

<thread>
{thread}
</thread>

{reference_outputs}
"""

    # Create judge model
    try:
        # Use load_chat_model directly as suggested
        custom_judge = load_chat_model(evaluator_model)

        logger.info(f"Created custom judge: {type(custom_judge)}")

    except Exception as e:
        logger.error(f"Failed to create custom judge: {e}")
        raise

    # Create the async LLM-as-judge evaluator with custom prompt
    evaluator = create_async_graph_trajectory_llm_as_judge(
        judge=custom_judge,
        prompt=CUSTOM_PROMPT,
        continuous=False,  # Use boolean scoring instead of continuous
        use_reasoning=True,  # Enable reasoning explanation in output
        feedback_key="react_agent_trajectory_quality",  # Custom feedback key
    )

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


# Pytest test functions
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", EvaluationConfig.AGENT_MODELS)
@pytest.mark.parametrize("scenario", EvaluationConfig.TEST_SCENARIOS)
async def test_graph_trajectory_evaluation(model_name, scenario):
    """Test graph trajectory evaluation with LLM-as-judge for all model-scenario combinations."""
    # Skip if required environment variables are not set
    required_env_vars = ["TAVILY_API_KEY", "SILICONFLOW_API_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            pytest.skip(f"Required environment variable {var} not set")

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

    # Run evaluation
    evaluation_result = await evaluate_trajectory(
        trajectory_data=trajectory_data,
        evaluator_model=EvaluationConfig.EVALUATOR_MODEL,
    )

    # Verify evaluation completed
    assert "evaluation" in evaluation_result or "evaluation_error" in evaluation_result

    # If evaluation succeeded, assert score is True
    if "evaluation" in evaluation_result:
        eval_data = evaluation_result["evaluation"]
        score = eval_data.get("score", False)
        reasoning = eval_data.get("reasoning", "No reasoning provided")

        logger.info(
            f"[{model_name}] [{scenario['name']}] Score: {score}, Reasoning: {reasoning}"
        )
        assert isinstance(score, bool), f"Expected boolean score, got {type(score)}"
        assert score is True, (
            f"Expected score to be True, got {score}. Reasoning: {reasoning}"
        )
