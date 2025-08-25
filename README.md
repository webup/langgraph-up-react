# LangGraph ReAct Agent Template

[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/webup/langgraph-up-react)
[![Build](https://github.com/webup/langgraph-up-react/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/webup/langgraph-up-react/actions/workflows/integration-tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![README EN](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![README CN](https://img.shields.io/badge/README-中文-red.svg)](./README_CN.md)
[![DeepWiki](https://img.shields.io/badge/Powered_by-DeepWiki-blue.svg)](https://deepwiki.com/webup/langgraph-up-react)
[![Twitter](https://img.shields.io/twitter/follow/zhanghaili0610?style=social)](https://twitter.com/zhanghaili0610)

This template showcases a [ReAct agent](https://arxiv.org/abs/2210.03629) implemented using [LangGraph](https://github.com/langchain-ai/langgraph), designed for [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). ReAct agents are uncomplicated, prototypical agents that can be flexibly extended to many tools.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

The core logic, defined in `src/react_agent/graph.py`, demonstrates a flexible ReAct agent that iteratively reasons about user queries and executes actions. The template features a modular architecture with shared components in `src/common/`, MCP integration for external documentation sources, and comprehensive testing suite.

**⭐ Star this repo if you find it helpful!** Visit our [webinar series](https://space.bilibili.com/31004924) for tutorials and advanced LangGraph development techniques.

## Features

### Dynamic Tool Loading
- **Web Search**: Built-in Tavily search integration
- **Documentation Tools**: DeepWiki MCP integration for GitHub repository documentation
- **Extensible**: Easy to add custom tools via `src/common/tools.py`

### Multi-Provider Model Support
- **OpenAI**: GPT-4o, GPT-4o-mini, etc.
  - **OpenAI-Compatible**: Any provider supporting OpenAI API format via custom API key and base URL
- **Anthropic**: Claude 4 Sonnet, Claude 3.5 Haiku, etc.
- **Qwen**: Qwen-Plus, Qwen-Turbo, QwQ-32B, QvQ-72B with regional API support

### MCP Integration
- **Model Context Protocol**: Dynamic external tool loading at runtime
- **DeepWiki MCP Server**: Repository documentation access and Q&A capabilities
- **Caching**: Optimized performance with client and tools caching
- **Configurable**: Enable via environment variables or context parameters

### Comprehensive Testing
- **70+ Test Cases**: Unit, integration, and end-to-end testing
- **MCP Integration Coverage**: Full testing of DeepWiki tool loading and execution
- **ReAct Loop Validation**: Tests verify proper tool-model interactions
- **Mock Support**: Reliable testing without external API calls

## What it Does

The ReAct agent:

1. Takes a user **query** as input
2. Reasons about the query and decides on an action
3. Executes the chosen action using available tools
4. Observes the result of the action
5. Repeats steps 2-4 until it can provide a final answer

The agent comes with web search capabilities and optional DeepWiki MCP documentation tools, but can be easily extended with custom tools to suit various use cases.

## Getting Started

### Setup with uv (Recommended)

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/webup/langgraph-up-react.git
cd langgraph-up-react
```

3. Install dependencies (including dev dependencies):
```bash
uv sync --dev
```

4. Copy the example environment file and fill in essential keys:
```bash
cp .env.example .env
```

### Environment Configuration

1. Edit the `.env` file with your API keys:

2. Define required API keys in your `.env` file:

```bash
# Required: Web search functionality
TAVILY_API_KEY=your-tavily-api-key

# Model providers (choose at least one)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DASHSCOPE_API_KEY=your-dashscope-api-key  # For Qwen models

# Optional: Regional API support for Qwen models
REGION=international  # or 'prc' for China mainland

# Optional: Enable documentation tools
ENABLE_DEEPWIKI=true
```

The primary [search tool](./src/common/tools.py) uses [Tavily](https://tavily.com/). Create an API key [here](https://app.tavily.com/sign-in).

## Model Setup

The default model configuration is:

```yaml
model: qwen:qwen-turbo
```

### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup)
2. Add it to your `.env` file:
```bash
OPENAI_API_KEY=your-api-key
```

### Anthropic

To use Anthropic's chat models:

1. Sign up for an [Anthropic API key](https://console.anthropic.com/)
2. Add it to your `.env` file:
```bash
ANTHROPIC_API_KEY=your-api-key
```
3. Update the model in LangGraph Studio to `anthropic:claude-3-5-sonnet-20240620`

### Qwen Models (Default)

For Alibaba's Qwen models (Qwen3, QwQ-32B, etc.):

1. Sign up for a [DashScope API key](https://dashscope.console.aliyun.com/)
2. Add it to your `.env` file:
```bash
DASHSCOPE_API_KEY=your-api-key
REGION=international  # or 'prc' for China mainland
```
3. Update the model in LangGraph Studio to `qwen:qwen3-32b` or `qwen:qwen-plus`

### OpenAI-Compatible Providers

For any OpenAI-compatible API (SiliconFlow, Together AI, Groq, etc.):

1. Get your API key from the provider
2. Add to your `.env` file:
```bash
# Example for custom OpenAI-compatible provider
OPENAI_API_KEY=your-provider-api-key
OPENAI_API_BASE=https://your-provider-api-base-url/v1
```
3. Update the model in LangGraph Studio to `openai:provider-model-name`

This flexible architecture allows you to use any OpenAI-compatible API by simply providing the API key and base URL.

## How to Customize

### Add New Tools
Extend the agent's capabilities by adding tools in [`src/common/tools.py`](./src/common/tools.py):

```python
async def my_custom_tool(input: str) -> str:
    """Your custom tool implementation."""
    return "Tool result"

# Add to the tools list in get_tools()
```

### Add New MCP Tools
Integrate external MCP servers for additional capabilities:

1. **Configure MCP Server** in [`src/common/mcp.py`](./src/common/mcp.py):
```python
MCP_SERVERS = {
    "deepwiki": {
        "url": "https://mcp.deepwiki.com/mcp",
        "transport": "streamable_http",
    },
    "your_mcp_server": {  # Add your MCP server
        "url": "https://your-mcp-server.com/mcp",
        "transport": "streamable_http",
    }
}
```

2. **Add Server Function**:
```python
async def get_your_mcp_tools() -> List[Callable[..., Any]]:
    """Get tools from your MCP server."""
    return await get_mcp_tools("your_mcp_server")
```

3. **Enable in Context** - Add context flag and load tools in `get_tools()` function.

### Configure Models
Our key extended method `load_chat_model` in [`src/common/utils.py`](./src/common/utils.py) uses LangChain's [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html#init-chat-model) as the underlying utility.

**Model String Format**: `provider:model-name` (follows LangChain's naming convention)

**Examples**:
```python
# OpenAI models
model = "openai:gpt-4o-mini"
model = "openai:gpt-4o"

# Qwen models (with regional support)
model = "qwen:qwen-turbo"    # Default model
model = "qwen:qwen-plus"     # Balanced performance
model = "qwen:qwq-32b-preview"  # Reasoning model
model = "qwen:qvq-72b-preview"  # Multimodal reasoning

# Anthropic models
model = "anthropic:claude-4-sonnet"
model = "anthropic:claude-3.5-haiku"
```

**Configuration**:
```bash
# Set via environment variable
MODEL=qwen:qwen-turbo

# Or in LangGraph Studio configurable settings
```

### Customize Prompts
Update the system prompt in [`src/common/prompts.py`](./src/common/prompts.py) or via the LangGraph Studio interface.

### Modify Agent Logic
Adjust the ReAct loop in [`src/react_agent/graph.py`](./src/react_agent/graph.py):
- Add new graph nodes
- Modify conditional routing logic  
- Add interrupts or human-in-the-loop interactions

### Configuration Options
Runtime configuration is managed in [`src/common/context.py`](./src/common/context.py):
- Model selection
- Search result limits
- Tool toggles

## Development

### Development Server
```bash
make dev        # Start LangGraph development server
make dev_ui     # Start with browser UI
```

### Testing
```bash
make test                    # Run unit and integration tests (default)
make test_unit               # Run unit tests only
make test_integration        # Run integration tests  
make test_e2e               # Run end-to-end tests (requires running server)
make test_all               # Run all test suites
```

### Code Quality
```bash
make lint       # Run linters (ruff + mypy)
make format     # Auto-format code
make lint_tests # Lint test files only
```

### Development Features
- **Hot Reload**: Local changes automatically applied
- **State Editing**: Edit past state and rerun from specific points
- **Thread Management**: Create new threads or continue existing conversations
- **LangSmith Integration**: Detailed tracing and collaboration

## Architecture

The template uses a modular architecture:

- **`src/react_agent/`**: Core agent graph and state management
- **`src/common/`**: Shared components (context, models, tools, prompts, MCP integration)
- **`tests/`**: Comprehensive test suite with fixtures and MCP integration coverage
- **`langgraph.json`**: LangGraph Studio configuration

Key components:
- **`src/common/mcp.py`**: MCP client management for external documentation sources
- **Dynamic tool loading**: Runtime tool selection based on context configuration
- **Context system**: Centralized configuration with environment variable support

This structure supports multiple agents and easy component reuse across different implementations.

## Development & Community

### Roadmap & Contributing
- 📋 **[ROADMAP.md](./ROADMAP.md)** - Current milestones and future plans
- 🐛 **Issues & PRs Welcome** - Help us improve by [raising issues](https://github.com/webup/langgraph-up-react/issues) or submitting pull requests
- 🤖 **Built with Claude Code** - This template is actively developed using [Claude Code](https://claude.ai/code)

### Getting Involved
We encourage community contributions! Whether it's:
- Reporting bugs or suggesting features
- Adding new tools or model integrations  
- Improving documentation
- Sharing your use cases and templates

Check out our roadmap to see what we're working on next and how you can contribute.

## Learn More

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph) - Framework guides and examples
- [LangSmith](https://smith.langchain.com/) - Tracing and collaboration platform  
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original research on reasoning and acting
- [Claude Code](https://claude.ai/code) - AI-powered development environment