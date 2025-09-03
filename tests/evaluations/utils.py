"""Utility functions for evaluation tests."""

import os
from typing import Dict, List, Optional


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
    return [name for name in essential_requirements if not requirements.get(name, False)]


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
                if hasattr(value, "model_dump"):  # LangChain message object (Pydantic v2)
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


def calculate_evaluation_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate aggregate metrics from evaluation results.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Dictionary of calculated metrics
    """
    if not results:
        return {}
    
    successful_results = [r for r in results if "evaluation" in r]
    
    if not successful_results:
        return {"success_rate": 0.0}
    
    # Extract scores (handle both numeric and boolean scores)
    scores = []
    for result in successful_results:
        eval_data = result["evaluation"]
        score = eval_data.get("score", 0)
        
        # Convert boolean to numeric
        if isinstance(score, bool):
            score = 1.0 if score else 0.0
        elif isinstance(score, str):
            # Try to parse numeric strings
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
        
        scores.append(float(score))
    
    return {
        "success_rate": len(successful_results) / len(results),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "total_tests": len(results),
        "successful_tests": len(successful_results),
        "failed_tests": len(results) - len(successful_results),
    }