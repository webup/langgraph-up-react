"""Graph trajectory evaluation using AgentEvals framework with LangSmith integration.

This module implements comprehensive evaluation of the ReAct agent using graph trajectory
LLM-as-judge methodology with async execution and OpenEvals aevaluate integration.

Usage:
    python graph.py                           # Run all models on all scenarios
    python graph.py --model siliconflow:Qwen/Qwen3-8B   # Test specific model on all scenarios
    python graph.py --verbose                           # Enable debug output
"""

import argparse
import asyncio
import logging
import os

# Import load_chat_model from common utils
import sys
import traceback
from typing import Any, Dict

from agentevals.graph_trajectory.llm import create_async_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import aextract_langgraph_trajectory_from_thread
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from langsmith.evaluation import aevaluate

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from config import EVALUATION_CONFIG
from utils import (
    GRAPH_TRAJECTORY_SCORE_NAMES,
    extract_scores_from_results,
    normalize_trajectory_data,
    print_evaluation_summary,
    print_scores_summary,
)

from src.common.context import Context
from src.common.utils import load_chat_model

# Load environment variables
load_dotenv()

# Configure logging for evaluation
logger = logging.getLogger(__name__)

# Initialize LangSmith client
langsmith_client = Client()

# Dataset configuration
DATASET_NAME = f"{EVALUATION_CONFIG['DATASET_PREFIX']}-graph-trajectory"


class EvaluationConfig:
    """Configuration for graph trajectory evaluation."""

    # Agent models to test - using shared configuration
    AGENT_MODELS = [
        EVALUATION_CONFIG["MODEL_AI"],  # Primary AI agent model
        EVALUATION_CONFIG["MODEL_PERSONA"],  # Alternative model for testing
    ]

    # Evaluator model - using shared configuration
    EVALUATOR_MODEL = EVALUATION_CONFIG["MODEL_EVALUATOR"]

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


async def ensure_dataset():
    """Ensure dataset exists, create only if it doesn't exist."""
    try:
        existing_datasets = langsmith_client.list_datasets()
        for dataset in existing_datasets:
            if dataset.name == DATASET_NAME:
                print(f"📋 Using existing dataset: {DATASET_NAME}")
                return DATASET_NAME
    except Exception:
        pass

    # Create dataset if it doesn't exist
    print(f"📋 Creating new dataset: {DATASET_NAME}")
    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Graph trajectory evaluation scenarios for ReAct agent testing with custom evaluator",
    )

    # Add examples for each scenario following the sample data format
    for scenario in EvaluationConfig.TEST_SCENARIOS:
        langsmith_client.create_example(
            inputs={
                "question": scenario["query"],
                "scenario": scenario[
                    "name"
                ],  # Use "scenario" instead of "scene" to match sample
            },
            outputs={"expected_behavior": scenario["expected_behavior"]},
            dataset_id=dataset.id,
            metadata={
                "scenario": scenario["name"],
                "query_type": scenario["name"],
                "dataset_split": ["base"],  # Add dataset_split to match sample format
            },
        )

    print(
        f"✨ Created dataset '{DATASET_NAME}' with {len(EvaluationConfig.TEST_SCENARIOS)} scenarios"
    )
    return DATASET_NAME


async def create_trajectory_app(model_name: str, verbose: bool = False):
    """Create trajectory evaluation app for a specific model."""

    async def app(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run trajectory evaluation for model-scenario combination."""
        # Extract from dict structure with question and scenario (updated from scene)
        question = inputs.get("question", "")
        scenario_name = inputs.get(
            "scenario", inputs.get("scene", "unknown")
        )  # Support both keys for compatibility

        if verbose:
            print(f"🔧 Running trajectory evaluation: {model_name} on {scenario_name}")
            print(f"📋 Query: {question[:100]}...")

        # Generate unique thread ID for this evaluation
        thread_id = (
            f"eval_{model_name.replace(':', '_').replace('/', '_')}_{scenario_name}"
        )

        # Run agent scenario
        trajectory_data = await run_agent_scenario(
            model_name=model_name, query=question, thread_id=thread_id
        )

        if verbose and "error" in trajectory_data:
            print(f"❌ Error: {trajectory_data['error']}")
        elif verbose:
            traj = trajectory_data.get("trajectory", {})
            steps_count = 0
            if "steps" in traj:
                steps_count = (
                    len(traj["steps"]) if isinstance(traj["steps"], list) else 0
                )
            print(f"✅ Trajectory captured: {steps_count} steps")
            print(
                f"🔍 Trajectory structure: {list(traj.keys()) if isinstance(traj, dict) else 'Not a dict'}"
            )
            if verbose and traj:
                print(f"🔍 Full trajectory keys: {list(traj.keys())}")
                if "outputs" in traj and "steps" in traj["outputs"]:
                    print(f"🔍 Steps in outputs: {traj['outputs']['steps']}")

        # Return trajectory outputs directly
        if "error" in trajectory_data:
            return {
                "error": trajectory_data["error"],
                "model": model_name,
                "scenario": scenario_name,
                "query": question,
            }

        # Extract result from trajectory data - maintaining sample format
        result = trajectory_data.get("result", {})

        # Return in format similar to sample data
        return {
            "trajectory_data": trajectory_data,
            "messages": result.get("messages", []),
            "model": model_name,
            "scenario": scenario_name,
            "query": question,
            "thread_id": thread_id,
        }

    return app


async def create_trajectory_evaluators():
    """Create custom evaluators for trajectory assessment with scenario-specific rubrics."""

    # Create scenario mapping for easy lookup
    scenarios = {
        scenario["name"]: scenario for scenario in EvaluationConfig.TEST_SCENARIOS
    }

    async def graph_trajectory_scenario_judge(run, example=None):
        """Graph trajectory evaluator with scenario-specific rubrics using direct judge approach."""
        try:
            # Extract scenario information from example
            scenario_name = None
            if example and hasattr(example, "inputs") and example.inputs:
                scenario_name = example.inputs.get(
                    "scenario", example.inputs.get("scene")
                )

            if not scenario_name or scenario_name not in scenarios:
                return {
                    "key": "graph_trajectory_accuracy",
                    "score": 0,
                    "comment": f"Unknown scenario: {scenario_name}",
                }

            # Get scenario configuration
            scenario = scenarios[scenario_name]

            # Quick check for obvious failures only
            if not run or not run.outputs:
                return {
                    "key": "graph_trajectory_accuracy",
                    "score": 0,
                    "comment": "No run outputs available - obvious failure",
                }

            # Get trajectory data directly from run outputs (already extracted by app)
            trajectory_data = run.outputs.get("trajectory_data", {})

            # Quick check for obvious execution failures
            if not trajectory_data or "error" in trajectory_data:
                return {
                    "key": "graph_trajectory_accuracy",
                    "score": 0,
                    "comment": f"Trajectory execution error - obvious failure: {trajectory_data.get('error', 'Unknown error')}",
                }

            # Extract and normalize trajectory with inputs and outputs
            extracted_trajectory = trajectory_data.get("trajectory", {})
            if not extracted_trajectory:
                return {
                    "key": "graph_trajectory_accuracy",
                    "score": 0,
                    "comment": "No trajectory data found",
                }

            # Normalize trajectory data to make it JSON serializable
            serializable_inputs, serializable_outputs = normalize_trajectory_data(
                extracted_trajectory
            )

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

            # Create graph trajectory judge with custom prompt
            evaluator = create_async_graph_trajectory_llm_as_judge(
                prompt=custom_prompt,
                judge=load_chat_model(EvaluationConfig.EVALUATOR_MODEL),
                continuous=False,
                use_reasoning=True,
                feedback_key="graph_trajectory_accuracy",
            )

            # Call judge directly with normalized inputs/outputs
            result = await evaluator(
                inputs=serializable_inputs,
                outputs=serializable_outputs,
            )

            # Add scenario context to result
            if isinstance(result, dict):
                result["scenario"] = scenario_name

            return result

        except Exception as e:
            return {
                "key": "graph_trajectory_accuracy",
                "score": 0,
                "comment": f"Evaluation error: {str(e)}",
            }

    return [graph_trajectory_scenario_judge]


async def run_model_experiment(
    model_name: str,
    dataset_name: str,
    verbose: bool = False,
):
    """Run evaluation experiment for a specific model across all scenarios in dataset."""
    # Use full model name as experiment name
    experiment_name = model_name

    print(f"🚀 Running experiment: {experiment_name}")
    print(f"🤖 Testing model '{model_name}' on dataset with all scenarios")

    # Create trajectory app for this model
    app = await create_trajectory_app(model_name, verbose)

    # Create evaluators for all scenarios
    evaluators = await create_trajectory_evaluators()

    # Run evaluation
    async_results = await aevaluate(
        app,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_name,
        metadata={
            "model": model_name,
            "model_ai": model_name,
            "model_evaluator": EvaluationConfig.EVALUATOR_MODEL,
            "evaluation_type": "graph_trajectory",
        },
    )

    # Collect results
    all_results = []

    async for result_row in async_results:
        all_results.append(result_row)

    # Extract scores using the utils function
    extracted_scores = extract_scores_from_results(all_results)

    # Print detailed score summary
    print_scores_summary(
        model_name=model_name,
        extracted_scores=extracted_scores,
        score_display_names=GRAPH_TRAJECTORY_SCORE_NAMES,
        score_scale=100.0,
        score_suffix="%",
    )

    return {
        "experiment_name": experiment_name,
        "results": all_results,
        "extracted_scores": extracted_scores,
        "total_results": len(all_results),
        "status": "completed",
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Graph Trajectory Evaluation with OpenEvals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=EvaluationConfig.AGENT_MODELS,
        help="Run evaluation for specific model only",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging"
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )

    return parser.parse_args()


async def main():
    """Run the graph trajectory evaluation."""
    args = parse_args()

    # Handle list options
    if args.list_models:
        print("🤖 Available Models:")
        for model in EvaluationConfig.AGENT_MODELS:
            print(f"  • {model}")
        return

    if args.list_scenarios:
        print("📋 Available Scenarios:")
        for scenario in EvaluationConfig.TEST_SCENARIOS:
            print(f"  • {scenario['name']}: {scenario['expected_behavior']}")
        return

    # Check required environment variables (skip LANGSMITH_API_KEY for testing)
    required_env_vars = ["TAVILY_API_KEY", "SILICONFLOW_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables and try again.")
        return

    print("🎯 Graph Trajectory Evaluation")
    print("=" * 50)
    print(f"🤖 AI Models: {', '.join(EvaluationConfig.AGENT_MODELS)}")
    print(f"⚖️  Evaluator Model: {EvaluationConfig.EVALUATOR_MODEL}")
    print(f"📋 Scenarios: {len(EvaluationConfig.TEST_SCENARIOS)}")
    if args.verbose:
        print("🔊 Verbose mode enabled")

    # Ensure dataset exists
    dataset_name = await ensure_dataset()

    # Determine which models to run (always run all scenarios from dataset)
    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = EvaluationConfig.AGENT_MODELS

    # Run experiments by model (each model gets one experiment with all scenarios from dataset)
    results = {}
    experiment_count = 0
    total_experiments = len(models_to_run)
    total_evaluations = len(models_to_run) * len(EvaluationConfig.TEST_SCENARIOS)

    print(
        f"\n🧪 Running {total_experiments} model experiments ({total_evaluations} total evaluations):"
    )

    for model_name in models_to_run:
        experiment_count += 1

        print(f"\n[{experiment_count}/{total_experiments}] Testing model: {model_name}")

        try:
            result = await run_model_experiment(model_name, dataset_name, args.verbose)
            results[model_name] = result

        except Exception as e:
            print(f"❌ Failed: {model_name} - {e}")
            traceback.print_exc()
            results[model_name] = {"status": "failed", "error": str(e)}

    # Use the utils function for comprehensive summary
    print_evaluation_summary(
        results=results,
        score_display_names=GRAPH_TRAJECTORY_SCORE_NAMES,
        score_scale=100.0,
        score_suffix="%",
    )


if __name__ == "__main__":
    asyncio.run(main())
