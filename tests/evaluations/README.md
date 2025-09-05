# Graph Trajectory Evaluation Tests

This directory contains evaluation tests for the ReAct agent using the AgentEvals framework with Graph trajectory LLM-as-judge methodology and LangSmith pytest integration.

## References

- [AgentEvals Graph Trajectory LLM-as-Judge](https://github.com/langchain-ai/agentevals/blob/main/README.md#graph-trajectory-llm-as-judge)
- [AgentEvals LangSmith Integration](https://github.com/langchain-ai/agentevals/blob/main/README.md#langsmith-integration)
- [LangSmith Pytest Documentation](https://docs.langchain.com/langsmith/pytest)

## Overview

The evaluation system tests the ReAct agent's performance across multiple scenarios using:

- **Agent Models**: 
  - `siliconflow:Qwen/Qwen3-8B` 
  - `siliconflow:THUDM/GLM-4-9B-0414`
  
- **Evaluator Model**: 
  - `siliconflow:THUDM/GLM-Z1-9B-0414` (advanced reasoning model for evaluation)

- **Evaluation Method**: 
  - [Graph trajectory LLM-as-judge](https://github.com/langchain-ai/agentevals/blob/main/README.md#graph-trajectory-llm-as-judge) with async execution
  - [LangSmith pytest integration](https://docs.langchain.com/langsmith/pytest) for comprehensive tracking
  - Scenario-specific rubrics and evaluation criteria
  - Real performance metrics instead of pass/fail assertions

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

# Required for LangSmith integration
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"

# Optional: Set region for SiliconFlow API
export REGION="prc"  # or "international"
```

## Running Evaluations

### Graph Evaluation Tests

Run the ReAct agent graph trajectory evaluation tests:

```bash
# Using Makefile (recommended) - Fast parallel execution
make test_eval_graph

# Or run directly with pytest
pytest -n auto tests/evaluations/graph.py -v
```

### Specific Test Filtering

Run specific scenarios or models:

```bash
# Run specific scenario
pytest -n auto tests/evaluations/graph.py -k "simple_question" -v

# Run specific model
pytest -n auto tests/evaluations/graph.py -k "Qwen3-8B" -v

# Run all combinations (default)
make test_eval_graph
```

This runs **2 models × 3 scenarios = 6 parameterized test combinations** in parallel (~1-2 minutes).

### LangSmith Integration (Optional)

For detailed evaluation tracking and analysis:

```bash
# Sequential execution with LangSmith dashboard
LANGSMITH_TRACING=true pytest tests/evaluations/graph.py --langsmith-output -v
```

**Note**: LangSmith integration requires sequential execution and takes longer (~4-5 minutes).

## Test Scenarios

The evaluation includes these test scenarios with scenario-specific rubrics:

1. **Simple Question** (`simple_question`)
   - **Query**: "What is the capital of France?"
   - **Expected**: Direct answer without unnecessary tool usage
   - **Rubric**: Evaluates efficiency and appropriate confidence for basic facts
   - **Example Results**:
     - ❌ **Fail**: [Agent used tools unnecessarily](https://smith.langchain.com/public/cde5921c-48fc-46a7-a8bb-6e8d31821a6f/r) - Trajectory shows `tools` node for basic factual question (Score: 0)
     - ✅ **Success**: [Agent answered directly](https://smith.langchain.com/public/a965ad02-d4ac-4c87-8cb1-8b717ba3ca97/r) - Trajectory shows only `call_model` without tools (Score: 1)

2. **Search Required** (`search_required`)
   - **Query**: "What's the latest news about artificial intelligence?"
   - **Expected**: Uses search tools to find current information
   - **Rubric**: Evaluates search tool usage and information synthesis
   - **Example Results**:
     - ❌ **Fail**: [Agent provided generic content with links](https://smith.langchain.com/public/5b796d70-cf73-441c-a278-ff9d2493ecf2/r) - Used tools but gave generic summaries and link lists instead of specific current news (Score: 0)
     - ✅ **Success**: [Agent synthesized actual current information](https://smith.langchain.com/public/708fb561-92f1-482a-aef4-f26df874822d/r) - Used tools and provided specific recent developments with concrete details (Score: 1)

3. **Multi-step Reasoning** (`multi_step_reasoning`)
   - **Query**: "What are the pros and cons of renewable energy, and what are the latest developments?"
   - **Expected**: Search for information and provide structured analysis
   - **Rubric**: Evaluates complex analytical tasks and comprehensive research
   - **Example Results**:
     - ✅ **Success**: [Agent performed search and analytical synthesis](https://smith.langchain.com/public/59157ed9-d185-4e3f-99dd-d898a18a4178/r) - Used tools to gather current information and provided structured pros/cons analysis with recent developments (Score: 1)
     - ❌ **Potential Failures**: Agents that provide only generic pros/cons without search, or use tools but lack structured analysis of current developments

## Evaluation Criteria

Each agent trajectory is evaluated using **scenario-specific rubrics** with LangSmith integration:

### Evaluation Approach
- **Scenario-specific evaluators**: Each test scenario has custom evaluation criteria
- **Async graph trajectory evaluation**: Uses [`create_async_graph_trajectory_llm_as_judge`](https://github.com/langchain-ai/agentevals/blob/main/README.md#graph-trajectory-llm-as-judge)
- **Real performance metrics**: Actual scores (0.0, 0.5, 1.0) instead of pass/fail assertions
- **LangSmith tracking**: All inputs, outputs, and evaluation results logged via [pytest integration](https://docs.langchain.com/langsmith/pytest)

### Scoring
- **Real evaluation scores**: Reflects actual agent performance
- **Detailed reasoning**: Explanation for each evaluation decision
- **No artificial assertions**: Tests don't fail based on evaluation scores
- **Comprehensive feedback**: Available in LangSmith dashboard for analysis

## Configuration

Modify evaluation parameters in `graph.py`:

- `AGENT_MODELS`: List of models to test as agents (currently 2 models)
- `EVALUATOR_MODEL`: Model to use as the LLM judge (`siliconflow:THUDM/GLM-Z1-9B-0414`)
- `TEST_SCENARIOS`: Test cases with queries, expected behaviors, and custom rubrics (currently 3 scenarios)

## Test Architecture

### Parameterized Testing
- Uses `@pytest.mark.parametrize` to create all model-scenario combinations
- **Total combinations**: 2 models × 3 scenarios = 6 unique tests
- Each combination runs exactly once with unique thread IDs

### Key Features
- **LangSmith Integration**: Full [pytest integration](https://docs.langchain.com/langsmith/pytest) with `@pytest.mark.langsmith`
- **Scenario-specific rubrics**: Custom evaluation criteria for each test scenario
- **Real performance metrics**: Actual evaluation scores instead of artificial assertions
- **Comprehensive logging**: Inputs, outputs, reference outputs, and evaluation feedback
- **Async execution**: Efficient concurrent evaluation with [AgentEvals graph trajectory approach](https://github.com/langchain-ai/agentevals/blob/main/README.md#graph-trajectory-llm-as-judge)

## Output

Evaluation results include:

### LangSmith Dashboard
- **LangSmith URL**: Generated for each test run with detailed analytics
- **Comprehensive tracking**: All test inputs, outputs, and evaluation feedback
- **Performance metrics**: Real evaluation scores and reasoning for analysis
- **Historical comparison**: Track performance over time across models and scenarios

### Test Output
- **Individual test results**: Real evaluation scores (0.0-1.0) with detailed reasoning
- **Test status**: All tests pass (no artificial score assertions)
- **Execution time**: Sequential ~4-5 minutes, Parallel ~1-2 minutes
- **Detailed logging**: Model names, scenarios, scores, and evaluation explanations
- **Input format**: Shows actual question text (e.g., "What is the capital of France?")

## Benefits of LangSmith Integration

- **Real Performance Metrics**: Evaluation scores reflect actual agent capabilities
- **Comprehensive Tracking**: All test data stored in LangSmith for detailed analysis
- **Scenario Customization**: Each test scenario has appropriate evaluation criteria
- **Historical Analysis**: Track performance trends across different model versions
- **No Artificial Assertions**: Tests focus on measurement rather than pass/fail

## Notes

- **LangSmith Required**: Tests require `LANGSMITH_API_KEY` environment variable
- **Environment validation**: Tests skip gracefully when API keys are missing
- **Async implementation**: Uses original async graph trajectory evaluators
- **Production ready**: All linting issues resolved, comprehensive evaluation system