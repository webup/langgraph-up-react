# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph ReAct (Reasoning and Action) Agent template that implements an iterative reasoning agent using LangGraph framework. The agent processes user queries through a cycle of reasoning, action execution, and observation.

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
- **Model Providers**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DASHSCOPE_API_KEY` (for Qwen models)
- **Optional**: `REGION` (set to `prc` or `international` for Qwen API endpoints)
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
- `src/common/models.py`: Custom model integrations (Qwen, QwQ, QvQ) with regional API support
- `src/common/prompts.py`: System prompt templates
- `src/common/utils.py`: Shared utility functions

### Configuration
- `langgraph.json`: LangGraph Studio configuration pointing to the main graph
- `.env`: Environment variables for API keys and configuration

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
- Main dependencies: LangGraph, LangChain, provider-specific packages, langchain-mcp-adapters
- Development tools: mypy, ruff, pytest
- Package structure supports both standalone and LangGraph template usage

## Development Guidelines

- **Research Tools**: Use `context7` and/or `deepwiki` to study unfamiliar projects or frameworks
- **Code Quality**: Always run `make lint` after completing tasks
- **Testing**: Comprehensive test suite includes unit, integration, and e2e tests with DeepWiki MCP integration coverage
- **MCP Integration**: DeepWiki tools are dynamically loaded when `enable_deepwiki=True` in context configuration