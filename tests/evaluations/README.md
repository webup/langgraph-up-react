# Graph Trajectory Evaluation Tests

This directory contains evaluation tests for the ReAct agent using the AgentEvals framework with Graph trajectory LLM-as-judge methodology.

## Overview

The evaluation system tests the ReAct agent's performance across multiple scenarios using:

- **Agent Models**: 
  - `siliconflow:Qwen/Qwen3-8B` 
  - `siliconflow:THUDM/GLM-4-9B-0414`
  
- **Evaluator Model**: 
  - `siliconflow:deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (expert data labeler)

- **Evaluation Method**: 
  - Graph trajectory LLM-as-judge with async execution
  - Boolean scoring (true/false) with expert data labeler prompt
  - Custom JSON output guardrails for structured evaluation

## Files

- `graph.py` - Main evaluation implementation with async graph trajectory evaluation
- `utils.py` - Utility functions for evaluation helpers and metrics
- `conftest.py` - Pytest configuration and fixtures for evaluation tests
- `README.md` - This documentation file

## Requirements

Before running evaluations, ensure you have the required environment variables set:

```bash
# Required for all evaluations
export TAVILY_API_KEY="your_tavily_api_key"
export SILICONFLOW_API_KEY="your_siliconflow_api_key"  # For both agents and evaluator

# Optional: Set region for SiliconFlow API
export REGION="prc"  # or "international"
```

## Running Evaluations

### Quick Test

Run a single evaluation test:

```bash
make test_evaluations
```

Or use pytest directly:

```bash
uv run python -m pytest tests/evaluations/ -v
```

### Comprehensive Evaluation

Run the full evaluation suite across all models and scenarios (6 total combinations):

```bash
uv run python -m pytest tests/evaluations/graph.py::test_graph_trajectory_evaluation -v
```

This runs **2 models × 3 scenarios = 6 parameterized test combinations**.

## Test Scenarios

The evaluation includes these test scenarios:

1. **Simple Question** - Direct factual queries that don't require tool usage
2. **Search Required** - Queries requiring web search for current information  
3. **Multi-step Reasoning** - Complex queries requiring both search and structured analysis

## Evaluation Criteria

Each agent trajectory is evaluated using the **expert data labeler** methodology:

### Rubric
An accurate trajectory:
- Makes logical sense between steps
- Shows clear progression  
- Is relatively efficient, though it does not need to be perfectly efficient
- Is semantically equivalent to the provided reference trajectory, if present

### Scoring
- **Boolean scoring**: `true` (passes evaluation) or `false` (fails evaluation)
- **JSON output**: `{"score": true/false, "reasoning": "explanation"}`
- **Assertion**: All tests must return `true` to pass

## Configuration

Modify evaluation parameters in `graph.py`:

- `AGENT_MODELS`: List of models to test as agents (currently 2 models)
- `EVALUATOR_MODEL`: Model to use as the LLM judge (DeepSeek R1)
- `TEST_SCENARIOS`: Test cases with queries and expected behaviors (currently 3 scenarios)

## Test Architecture

### Parameterized Testing
- Uses `@pytest.mark.parametrize` to create all model-scenario combinations
- **Total combinations**: 2 models × 3 scenarios = 6 unique tests
- Each combination runs exactly once with unique thread IDs

### Key Features
- **Custom prompt**: Expert data labeler with JSON output guardrails
- **Boolean assertions**: Each evaluation must return `true` to pass
- **Trajectory normalization**: Handles LangChain message serialization 
- **Error handling**: Graceful handling of API failures and missing keys
- **Async execution**: Efficient concurrent evaluation

## Output

Evaluation results include:

- **Individual test results**: Boolean pass/fail with reasoning for each model-scenario combination
- **Pytest summary**: Clear pass/fail status for all 6 combinations
- **Execution time**: Total time for complete evaluation suite
- **Detailed logging**: Model names, scenarios, scores, and reasoning text

## Notes

- **No global test skipping**: Individual tests check their required environment variables
- **Environment validation**: Tests skip gracefully when API keys are missing
- **Async implementation**: Efficient execution across multiple model-scenario combinations  
- **Production ready**: All linting issues resolved, clean codebase