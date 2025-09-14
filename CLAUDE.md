# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph ReAct (Reasoning and Action) Agent template that implements an iterative reasoning agent using LangGraph framework. The agent processes user queries through a cycle of reasoning, action execution, and observation. **Version 0.2.0** introduces comprehensive evaluation systems and expanded model provider support.

## Architecture

The core architecture follows a modular stateful graph pattern:

- **Common Module**: Shared components in `src/common/` provide reusable functionality across agents
- **State Management**: Uses `State` and `InputState` dataclasses (defined in `src/react_agent/state.py`) to track conversation messages and execution state
- **Graph Structure**: The main graph is defined in `src/react_agent/graph.py` with two primary nodes:
  - `call_model`: Handles LLM reasoning and tool selection
  - `tools`: Executes selected tools via ToolNode
- **Execution Flow**: `call_model` → conditional routing → either `tools` (if tool calls needed) or `__end__` (if ready to respond) → back to `call_model` (creates the ReAct loop)
- **Context System**: Runtime context defined in `src/common/context.py` provides model configuration, system prompts, and DeepWiki integration control
- **Dynamic Tools**: Runtime tool loading with MCP integration for external documentation sources (DeepWiki MCP server)
- **Model Providers**: Multi-provider support including Anthropic, OpenAI, Qwen, QwQ, QvQ, and SiliconFlow with regional endpoint configuration

## Development Commands

### Testing
```bash
# Run specific test types
make test                    # Run unit and integration tests (default)
make test_unit               # Run unit tests only
make test_integration        # Run integration tests only
make test_e2e               # Run e2e tests only (requires running LangGraph server)
make test_all               # Run all tests (unit + integration + e2e)

# Watch mode for continuous testing
make test_watch             # Run unit tests in watch mode
make test_watch_unit        # Run unit tests in watch mode
make test_watch_integration # Run integration tests in watch mode
make test_watch_e2e         # Run e2e tests in watch mode

# Additional test options
make extended_tests         # Run extended test suite
```

### Evaluations
```bash
# Comprehensive evaluation suite (NEW in v0.2.0)
make evals                   # Run all evaluations (graph + multiturn)
make eval_graph              # Run graph trajectory evaluations (LLM-as-judge)
make eval_multiturn          # Run multi-turn chat evaluations (role-persona simulations)

# Model-specific evaluations
make eval_graph_qwen         # Test with Qwen/Qwen3-8B model
make eval_graph_glm          # Test with THUDM/GLM-4-9B model

# Persona-specific evaluations
make eval_multiturn_polite   # Test with polite persona
make eval_multiturn_hacker   # Test with hacker persona
```

### Code Quality
```bash
make lint                   # Run linters (ruff + mypy)  
make format                 # Auto-format code with ruff
make lint_package           # Lint only src/ directory
make lint_tests             # Lint only tests/ directory
```

### Development Server
```bash
make dev                    # Start LangGraph development server
make dev_ui                 # Start LangGraph development server with UI
```

### Environment Setup
- Copy `.env.example` to `.env` and configure API keys
- **Required**: `TAVILY_API_KEY` for web search functionality
- **Model Providers**:
  - `ANTHROPIC_API_KEY` for Anthropic models
  - `OPENAI_API_KEY` for OpenAI models
  - `DASHSCOPE_API_KEY` for Qwen/QwQ/QvQ models
  - `SILICONFLOW_API_KEY` for SiliconFlow models (NEW in v0.2.0)
- **Optional**: `REGION` (set to `prc`/`cn` or `international`/`en` for regional API endpoints)
- **Optional**: `ENABLE_DEEPWIKI=true` to enable DeepWiki MCP documentation tools
- **Default Model**: Uses `qwen:qwen-flash` as default model (configurable via `MODEL` environment variable)

## Key Files and Their Purposes

### Core Agent Files
- `src/react_agent/graph.py`: Main graph definition and ReAct loop implementation
- `src/react_agent/state.py`: State management with message tracking via `add_messages` annotation

### Common Module (Shared Components)
- `src/common/context.py`: Runtime context and configuration with environment variable support and DeepWiki integration
- `src/common/tools.py`: Tool definitions including web search and dynamic MCP tool loading
- `src/common/mcp.py`: MCP client management for external documentation sources (e.g. DeepWiki)
- `src/common/models/`: Model provider integrations with regional API support
  - `qwen.py`: Qwen, QwQ, QvQ model integrations via DashScope
  - `siliconflow.py`: SiliconFlow model integrations (NEW in v0.2.0)
- `src/common/prompts.py`: System prompt templates
- `src/common/utils.py`: Shared utility functions including model loading

### Configuration
- `langgraph.json`: LangGraph Studio configuration pointing to the main graph
- `.env`: Environment variables for API keys and configuration

### Evaluation System (NEW in v0.2.0)
- `tests/evaluations/`: Comprehensive evaluation framework
  - `graph.py`: Graph trajectory evaluation using AgentEvals with LLM-as-judge methodology
  - `multiturn.py`: Multi-turn chat evaluation with role-persona simulations
  - `config.py`: Evaluation configuration and test scenarios
  - `utils.py`: Shared evaluation utilities and scoring functions

## LangGraph Studio Integration

This project works seamlessly with LangGraph Studio. The `langgraph.json` config file defines:
- Graph entry point: `./src/react_agent/graph.py:graph`
- Environment file: `.env`
- Dependencies: current directory (`.`)

## MCP Integration

This project integrates with Model Context Protocol (MCP) servers for dynamic external tool loading:

- **DeepWiki MCP Server**: Provides documentation tools for GitHub repositories
- **Dynamic Loading**: Tools are loaded at runtime based on context configuration
- **Caching**: MCP client and tools are cached for performance
- **Configuration**: Enable via `ENABLE_DEEPWIKI=true` environment variable or context parameter
- **Server Configuration**: MCP servers configured in `src/common/mcp.py`

### MCP Usage
- Set `enable_deepwiki=True` in Context to load DeepWiki tools
- Tools include repository documentation access and Q&A capabilities
- Integrated seamlessly with LangGraph ReAct pattern for tool execution

## Python Configuration

- Python requirement: `>=3.11,<4.0`
- Main dependencies: LangGraph, LangChain, provider-specific packages, langchain-mcp-adapters, langchain-siliconflow
- Development tools: mypy, ruff, pytest, langgraph-cli, langgraph-sdk
- Evaluation dependencies: openevals, agentevals, langsmith (NEW in v0.2.0)
- Package structure supports both standalone and LangGraph template usage

## Development Guidelines

- **Research Tools**: Use `context7` and/or `deepwiki` to study unfamiliar projects or frameworks
- **Code Quality**: Always run `make lint` after completing tasks
- **Testing**: Comprehensive test suite includes unit, integration, and e2e tests with DeepWiki MCP integration coverage
- **Evaluation**: Use `make evals` to run comprehensive agent evaluations (NEW in v0.2.0)
  - Graph trajectory evaluation with LLM-as-judge methodology
  - Multi-turn conversation evaluation with role-persona simulations
  - Model-specific testing across different providers (Qwen, GLM, etc.)
- **MCP Integration**: DeepWiki tools are dynamically loaded when `enable_deepwiki=True` in context configuration
- **Multi-Model Support**: Test across different providers using evaluation commands for comprehensive coverage