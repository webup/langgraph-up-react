"""Utility functions for evaluation tests."""

import os
from typing import Dict, List, Optional

import httpx
from config import EVALUATION_CONFIG


def check_evaluation_requirements() -> Dict[str, bool]:
    """Check if required environment variables are set for evaluation.

    Returns:
        Dictionary mapping requirement names to availability status
    """
    return {
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "SILICONFLOW_API_KEY": bool(os.getenv("SILICONFLOW_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


def get_missing_requirements() -> List[str]:
    """Get list of missing required environment variables.

    Returns:
        List of missing environment variable names
    """
    requirements = check_evaluation_requirements()
    # Only check the essential requirements for our evaluation
    essential_requirements = ["TAVILY_API_KEY", "SILICONFLOW_API_KEY"]
    return [
        name for name in essential_requirements if not requirements.get(name, False)
    ]


def skip_if_missing_requirements(required_vars: Optional[List[str]] = None):
    """Decorator to skip tests if required environment variables are missing.

    Args:
        required_vars: List of required environment variable names.
                      If None, checks all standard requirements.
    """
    import pytest

    if required_vars is None:
        required_vars = ["TAVILY_API_KEY", "SILICONFLOW_API_KEY"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        return pytest.mark.skip(
            reason=f"Required environment variables not set: {', '.join(missing_vars)}"
        )

    return lambda func: func


async def check_langgraph_server(url: str = EVALUATION_CONFIG["LANGGRAPH_URL"]) -> bool:
    """Check if LangGraph server is running and accessible.

    Args:
        url: LangGraph server URL to check

    Returns:
        True if server is accessible, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/ok")
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    except Exception:
        return False


async def require_langgraph_server(url: str = EVALUATION_CONFIG["LANGGRAPH_URL"]):
    """Ensure LangGraph server is running before multiturn evaluation.

    Args:
        url: LangGraph server URL to check

    Raises:
        SystemExit: If server is not accessible with instructions
    """
    if not await check_langgraph_server(url):
        print("❌ LangGraph server is not running!")
        print("\n💡 Multi-turn evaluation requires the LangGraph development server.")
        print("   Please start it in a separate terminal:")
        print("\n   make dev")
        print("   # OR")
        print("   uv run langgraph dev --no-browser")
        print("\n   Then run the evaluation again.")
        raise SystemExit(1)

    print("✅ LangGraph server is running")


def format_model_name_for_filename(model_name: str) -> str:
    """Format model name to be safe for filenames.

    Args:
        model_name: Original model name (e.g., 'siliconflow:Qwen/Qwen3-8B')

    Returns:
        Filename-safe version (e.g., 'siliconflow_Qwen_Qwen3-8B')
    """
    return model_name.replace(":", "_").replace("/", "_").replace("-", "_")


def normalize_trajectory_data(trajectory: Dict) -> tuple:
    """Normalize trajectory data for AgentEvals input/output format.

    Args:
        trajectory: Raw trajectory data from LangGraph

    Returns:
        Tuple of (serializable_inputs, serializable_outputs)
    """
    # Convert inputs to serializable format
    serializable_inputs = []
    for inp in trajectory.get("inputs", []):
        if isinstance(inp, dict):
            serialized_inp = {}
            for key, value in inp.items():
                if hasattr(
                    value, "model_dump"
                ):  # LangChain message object (Pydantic v2)
                    serialized_inp[key] = value.model_dump()
                elif hasattr(value, "dict"):  # LangChain message object (Pydantic v1)
                    serialized_inp[key] = value.dict()
                elif isinstance(value, dict) and "messages" in value:
                    # Handle nested message structures
                    serialized_messages = []
                    for msg in value["messages"]:
                        if hasattr(msg, "model_dump"):  # Pydantic v2
                            serialized_messages.append(msg.model_dump())
                        elif hasattr(msg, "dict"):  # Pydantic v1
                            serialized_messages.append(msg.dict())
                        else:
                            serialized_messages.append(msg)
                    serialized_inp[key] = {"messages": serialized_messages}
                else:
                    serialized_inp[key] = value
            serializable_inputs.append(serialized_inp)
        else:
            serializable_inputs.append(inp)

    # Convert outputs to serializable format
    serializable_outputs = {}
    for key, value in trajectory.get("outputs", {}).items():
        if key == "results":
            serialized_results = []
            for result in value:
                if isinstance(result, dict) and "messages" in result:
                    serialized_messages = []
                    for msg in result["messages"]:
                        if hasattr(msg, "model_dump"):  # Pydantic v2
                            serialized_messages.append(msg.model_dump())
                        elif hasattr(msg, "dict"):  # Pydantic v1
                            serialized_messages.append(msg.dict())
                        else:
                            serialized_messages.append(msg)
                    serialized_results.append({"messages": serialized_messages})
                else:
                    serialized_results.append(result)
            serializable_outputs[key] = serialized_results
        else:
            serializable_outputs[key] = value

    return serializable_inputs, serializable_outputs


def extract_scores_from_results(result_rows: List[Dict]) -> List[Dict[str, any]]:
    """Extract evaluation scores from ExperimentResultRow objects.

    Args:
        result_rows: List of ExperimentResultRow objects from LangSmith evaluation

    Returns:
        List of dictionaries containing scores for each result
    """
    extracted_scores = []

    for result_row in result_rows:
        # Extract scores from evaluation_results
        scores = {}
        run_id = "unknown"

        # Handle both dict and object result_row structures
        evaluation_results = None
        if isinstance(result_row, dict):
            evaluation_results = result_row.get("evaluation_results", {})
            run_obj = result_row.get("run")
            if run_obj and hasattr(run_obj, "id"):
                run_id = str(run_obj.id)
        else:
            # Handle object-style access
            evaluation_results = getattr(result_row, "evaluation_results", {})
            run_obj = getattr(result_row, "run", None)
            if run_obj and hasattr(run_obj, "id"):
                run_id = str(run_obj.id)

        if evaluation_results:
            # Check if it has a 'results' key (LangSmith structure)
            if isinstance(evaluation_results, dict) and "results" in evaluation_results:
                eval_results_list = evaluation_results["results"]
                for eval_result in eval_results_list:
                    key = getattr(eval_result, "key", None)
                    score = getattr(eval_result, "score", None)
                    if key and score is not None:
                        scores[key] = score
            # Handle hasattr pattern for object access
            elif hasattr(evaluation_results, "results"):
                eval_results_list = evaluation_results.results
                for eval_result in eval_results_list:
                    key = getattr(eval_result, "key", None)
                    score = getattr(eval_result, "score", None)
                    if key and score is not None:
                        scores[key] = score
            # Handle direct dict mapping (alternative structure)
            elif isinstance(evaluation_results, dict):
                for key, eval_result in evaluation_results.items():
                    if key != "results":  # Skip the 'results' key if it exists
                        score = getattr(eval_result, "score", None)
                        if score is not None:
                            scores[key] = score

        extracted_scores.append({"run_id": run_id, "scores": scores})

    return extracted_scores


def print_scores_summary(
    model_name: str,
    extracted_scores: List[Dict[str, any]],
    score_display_names: Optional[Dict[str, str]] = None,
    score_scale: float = 1.0,
    score_suffix: str = "",
) -> None:
    """Print a formatted summary of evaluation scores.

    Args:
        model_name: Name of the model being evaluated
        extracted_scores: List of score dictionaries from extract_scores_from_results
        score_display_names: Optional mapping of internal keys to display names
        score_scale: Scale to multiply scores by (e.g., 100 for percentage)
        score_suffix: Suffix to add to scores (e.g., "%" for percentage)
    """
    if not extracted_scores:
        print(f"   ⚠️  No scores available for {model_name}")
        return

    print(f"   📊 Evaluation Scores for {model_name}:")

    # Calculate averages across all runs
    all_scores = {}
    for score_data in extracted_scores:
        for key, score in score_data["scores"].items():
            if key not in all_scores:
                all_scores[key] = []
            all_scores[key].append(score)

    # Use default display names if not provided
    if score_display_names is None:
        score_display_names = {}

    # Print individual run scores
    for i, score_data in enumerate(extracted_scores, 1):
        scores = score_data["scores"]
        print(f"     Run {i}:")
        for key, score in scores.items():
            display_name = score_display_names.get(key, key.replace("_", " ").title())
            formatted_score = score * score_scale
            print(f"       {display_name}: {formatted_score:.3f}{score_suffix}")

    # Print averages if multiple runs
    if len(extracted_scores) > 1:
        print("     Average Scores:")
        for key, scores_list in all_scores.items():
            if scores_list:
                avg_score = sum(scores_list) / len(scores_list)
                display_name = score_display_names.get(
                    key, key.replace("_", " ").title()
                )
                formatted_score = avg_score * score_scale
                print(f"       {display_name}: {formatted_score:.3f}{score_suffix}")


def print_evaluation_summary(
    results: Dict[str, Dict],
    score_display_names: Optional[Dict[str, str]] = None,
    score_scale: float = 100.0,
    score_suffix: str = "%",
) -> None:
    """Print overall evaluation summary with rankings.

    Args:
        results: Dictionary containing evaluation results for all models
        score_display_names: Optional mapping of internal keys to display names
        score_scale: Scale to multiply scores by (e.g., 100 for percentage)
        score_suffix: Suffix to add to scores (e.g., "%" for percentage)
    """
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    # Filter successful results
    successful_results = {
        k: v for k, v in results.items() if v and v.get("status") != "failed"
    }

    if not successful_results:
        print("❌ No successful evaluations to summarize")
        return

    # Collect all scores for ranking
    all_model_scores = {}

    for model_name, result_data in successful_results.items():
        extracted_scores = result_data.get("extracted_scores", [])

        # Calculate average scores for this model
        model_averages = {}
        for score_data in extracted_scores:
            for key, score in score_data["scores"].items():
                if key not in model_averages:
                    model_averages[key] = []
                model_averages[key].append(score)

        # Store averages for ranking
        for key, scores_list in model_averages.items():
            if scores_list:
                avg = sum(scores_list) / len(scores_list)
                if key not in all_model_scores:
                    all_model_scores[key] = []
                all_model_scores[key].append((model_name, avg))

    # Use default display names if not provided
    if score_display_names is None:
        score_display_names = {"graph_trajectory_accuracy": "Graph Trajectory Accuracy"}

    # Print rankings for each metric
    for metric_key, scores_with_models in all_model_scores.items():
        if scores_with_models:
            display_name = score_display_names.get(
                metric_key, metric_key.replace("_", " ").title()
            )
            print(f"\n🏆 {display_name} Rankings:")

            # Sort by score (highest first)
            sorted_scores = sorted(scores_with_models, key=lambda x: x[1], reverse=True)

            for rank, (model, score) in enumerate(sorted_scores, 1):
                medal = (
                    "🥇"
                    if rank == 1
                    else "🥈"
                    if rank == 2
                    else "🥉"
                    if rank == 3
                    else "  "
                )
                formatted_score = score * score_scale
                print(
                    f"   {medal} {rank}. {model}: {formatted_score:.1f}{score_suffix}"
                )

    # Print completion summary
    total_models = len(results)
    completed_models = len(successful_results)
    print(f"\n🎉 Completed: {completed_models}/{total_models} model evaluations")


# Display name mappings for graph trajectory evaluation
GRAPH_TRAJECTORY_SCORE_NAMES = {
    "graph_trajectory_accuracy": "Graph Trajectory Accuracy"
}

# Display name mappings for multiturn evaluation
MULTITURN_SCORE_NAMES = {
    "helpful": "Helpfulness",
    "progressive": "Progressive Conversation",
    "secure": "Security & Boundaries",
}
