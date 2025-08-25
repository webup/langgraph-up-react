# LangGraph ReAct Agent 模板

[![Version](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/webup/langgraph-up-react)
[![Build](https://github.com/webup/langgraph-up-react/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/webup/langgraph-up-react/actions/workflows/unit-tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![README EN](https://img.shields.io/badge/README-English-blue.svg)](./README.md)
[![README CN](https://img.shields.io/badge/README-中文-red.svg)](./README_CN.md)
[![DeepWiki](https://img.shields.io/badge/Powered_by-DeepWiki-blue.svg)](https://deepwiki.com/webup/langgraph-up-react)
[![Twitter](https://img.shields.io/twitter/follow/zhanghaili0610?style=social)](https://twitter.com/zhanghaili0610)

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的 [ReAct 智能体](https://arxiv.org/abs/2210.03629) 模板，专为本地开发者设计，支持 [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio)。ReAct 智能体是简洁的原型智能体，可以灵活扩展支持多种工具。

![LangGraph Studio 界面截图](./static/studio_ui.png)

核心逻辑定义在 `src/react_agent/graph.py` 中，展示了一个灵活的 ReAct 智能体，能够迭代地推理用户查询并执行操作。模板采用模块化架构，在 `src/common/` 中包含共享组件，集成 MCP 外部文档源，并提供完整的测试套件。

## 🌟 欢迎加入社区

**如果此项目对您有帮助，请点个 ⭐ Star，万分感谢！** 同时，您还可以访问我们的 [B 站空间](https://space.bilibili.com/31004924) 获取教程和 LangGraph 高级开发技巧。

### 📚 LangChain 实战系列图书

掌握 Agent 技术先机！从掌握 LangGraph 开始！我们的新书《LangGraph实战》现已出版，[点击查看详情](#langgraph实战书籍) ❤️

### 📱 加入飞书群

欢迎您扫描下方二维码加入我们的技术讨论群：

![飞书群二维码](./static/feishu.jpg)

## v0.1.0 核心特性

### 🚀 专为本地开发者设计
- **通义千问系列模型**: 通过 langchain-qwq 包提供完整的 Qwen 模型支持，包括 Qwen-Plus、Qwen-Turbo、QwQ-32B、QvQ-72B
- **OpenAI 兼容**: 支持 GPT-4o、GPT-4o-mini 等模型，以及任何 OpenAI API 格式的提供商

### 🛠 MCP 工具生态系统
- **模型上下文协议**: 运行时动态外部工具加载
- **DeepWiki MCP 服务器**: GitHub 仓库文档访问和问答功能
- **Web 搜索**: 内置 Tavily 搜索集成
- **可扩展**: 通过 `src/common/tools.py` 添加自定义工具

### 📊 LangGraph 平台开发支持
- **本地开发服务器**: 完整的 LangGraph Platform 开发环境
- **70+ 测试用例**: 单元、集成和端到端测试覆盖，完整测试 DeepWiki 工具加载和执行
- **ReAct 循环验证**: 确保正确的工具-模型交互

### 🤖 AI 驱动开发
- **使用 Claude Code 开发**: 本模板使用先进的 AI 开发环境构建
- **持续迭代**: 我们将持续完善和添加新模板
- **社区驱动**: 为不同场景的 LangGraph 开发提供更多选择

## 工作原理

ReAct 智能体的工作流程：

1. 接收用户**查询**作为输入
2. **推理**需要执行什么操作来回答查询
3. 使用可用工具执行选定的**操作**
4. **观察**操作结果
5. 重复步骤 2-4 直到能够提供最终答案

智能体内置 Web 搜索功能和可选的 DeepWiki MCP 文档工具，可轻松扩展以支持各种用例的自定义工具。

## 快速开始

### 使用 uv 安装（推荐）

1. 安装 uv（如果尚未安装）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. 克隆仓库：
```bash
git clone https://github.com/webup/langgraph-up-react.git
cd langgraph-up-react
```

3. 安装依赖（包括开发依赖）：
```bash
uv sync --dev
```

4. 复制示例环境文件并填入必要的 API 密钥：
```bash
cp .env.example .env
```

### 环境配置

1. 编辑 `.env` 文件，添加您的 API 密钥：

```bash
# 必需：搜索功能所需
TAVILY_API_KEY=your_tavily_api_key

# 可选：通义千问模型（默认）
DASHSCOPE_API_KEY=your_dashscope_api_key
REGION=prc  # 或 'international' 使用国内端点

# 可选：OpenAI 模型
OPENAI_API_KEY=your_openai_key

# 可选：启用 DeepWiki 文档工具
ENABLE_DEEPWIKI=true
```

主要搜索工具使用 [Tavily](https://tavily.com/)。在 [此处](https://app.tavily.com/sign-in) 创建 API 密钥。

## 模型配置

默认模型配置：

```yaml
model: qwen:qwen-turbo
```

### 通义千问（推荐用于本地开发）

通义千问模型提供强大的中文支持和高性价比：

1. 在 [DashScope 控制台](https://dashscope.console.aliyun.com/) 获取 API 密钥
2. 设置环境变量：

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key
REGION=prc  # 或 'international'
```

3. 在 LangGraph Studio 中使用以下模型之一：
   - `qwen:qwen-turbo` (快速，默认)
   - `qwen:qwen-plus` (平衡性能)
   - `qwen:qwq-32b-preview` (推理模型)

### OpenAI

使用 OpenAI 聊天模型：

1. 设置 API 密钥：
```bash
OPENAI_API_KEY=your_api_key
```

2. 在 LangGraph Studio 中更新模型为 `openai:gpt-4o-mini`

### OpenAI 兼容提供商

支持任何 OpenAI 兼容的 API：

1. 设置 API 密钥和基础 URL：
```bash
OPENAI_API_KEY=your_provider_api_key
OPENAI_API_BASE=https://your-provider-api-base-url/v1
```
2. 在 LangGraph Studio 中更新模型为 `openai:provider-model-name`

这种灵活的架构允许您通过简单提供 API 密钥和基础 URL 使用任何 OpenAI 兼容的 API。

## 自定义说明

### 添加新工具
在 [`src/common/tools.py`](./src/common/tools.py) 中扩展智能体功能：

```python
async def my_custom_tool(input: str) -> str:
    """自定义工具描述。"""
    # 您的工具逻辑
    return "工具输出"
```

### 添加新的 MCP 工具
集成外部 MCP 服务器以获得更多功能：

1. **配置 MCP 服务器**，在 [`src/common/mcp.py`](./src/common/mcp.py) 中：
```python
MCP_SERVERS = {
    "deepwiki": {
        "url": "https://mcp.deepwiki.com/mcp",
        "transport": "streamable_http",
    },
    "your_mcp_server": {  # 添加您的 MCP 服务器
        "url": "https://your-mcp-server.com/mcp",
        "transport": "streamable_http",
    }
}
```

2. **添加服务器函数**：
```python
async def get_your_mcp_tools() -> List[Callable[..., Any]]:
    """从您的 MCP 服务器获取工具。"""
    return await get_mcp_tools("your_mcp_server")
```

3. **在上下文中启用** - 添加上下文标志并在 `get_tools()` 函数中加载工具。

### 配置模型
我们在 [`src/common/utils.py`](./src/common/utils.py) 中的关键扩展方法 `load_chat_model` 使用 LangChain 的 [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html#init-chat-model) 作为底层工具。

**模型字符串格式**：`提供商:模型名称`（遵循 LangChain 命名约定）

**示例**：
```python
# OpenAI 模型
model = "openai:gpt-4o-mini"
model = "openai:gpt-4o"

# 通义千问模型（支持区域配置）
model = "qwen:qwen-turbo"    # 默认模型
model = "qwen:qwen-plus"     # 平衡性能
model = "qwen:qwq-32b-preview"  # 推理模型
model = "qwen:qvq-72b-preview"  # 多模态推理

# Anthropic 模型
model = "anthropic:claude-4-sonnet"
model = "anthropic:claude-3.5-haiku"
```

**配置方式**：
```bash
# 通过环境变量设置
MODEL=qwen:qwen-turbo

# 或在 LangGraph Studio 可配置设置中
```

### 修改系统提示
编辑 [`src/common/prompts.py`](./src/common/prompts.py) 中的默认系统提示。

### 更改模型
在 LangGraph Studio 中或通过环境变量更新 `model` 配置。

## 开发工作流

### 启动开发服务器
```bash
make dev        # 启动 LangGraph 开发服务器
make dev_ui     # 启动带浏览器界面的服务器
```

### 测试
```bash
make test                    # 运行单元和集成测试（默认）
make test_unit               # 仅运行单元测试
make test_integration        # 运行集成测试  
make test_e2e               # 运行端到端测试（需要运行服务器）
make test_all               # 运行所有测试套件
```

### 代码质量
```bash
make lint       # 运行 linter（ruff + mypy）
make format     # 使用 ruff 自动格式化代码
```

### LangGraph Studio 功能
- **可视化图结构**: 查看智能体的执行流程
- **实时调试**: 逐步执行和状态检查
- **交互式测试**: 直接在 Studio 中测试智能体

## 架构

模板采用模块化架构：

- **`src/react_agent/`**: 核心智能体图和状态管理
- **`src/common/`**: 共享组件（上下文、模型、工具、提示、MCP 集成）
- **`tests/`**: 完整测试套件，包含 fixtures 和 MCP 集成覆盖
- **`langgraph.json`**: LangGraph Studio 配置

关键组件：
- **`src/common/mcp.py`**: 外部文档源的 MCP 客户端管理
- **动态工具加载**: 基于上下文配置的运行时工具选择
- **上下文系统**: 支持环境变量的集中化配置

此架构支持多智能体和不同实现间的轻松组件重用。


## 开发与社区

### 路线图与贡献
- 📋 **[ROADMAP.md](./ROADMAP.md)** - 当前里程碑和未来规划
- 🐛 **欢迎 Issues 和 PR** - 通过 [提交 issue](https://github.com/webup/langgraph-up-react/issues) 或提交 pull request 帮助我们改进
- 🤖 **使用 Claude Code 开发** - 本模板使用 [Claude Code](https://claude.ai/code) 积极开发维护

### 参与贡献
我们鼓励社区贡献！无论是：
- 报告 bug 或建议新功能
- 添加新工具或模型集成  
- 改进文档
- 分享您的用例和模板

查看我们的路线图，了解下一步工作计划以及如何参与贡献。

## 了解更多

- [LangGraph 文档](https://github.com/langchain-ai/langgraph) - 框架指南和示例
- [通义千问模型文档](https://help.aliyun.com/zh/dashscope/) - 模型 API 和使用指南
- [MCP 协议](https://modelcontextprotocol.io/) - 了解模型上下文协议
- [ReAct 论文](https://arxiv.org/abs/2210.03629) - 原始研究论文
- [Claude Code](https://claude.ai/code) - AI 驱动的开发环境

## LangChain 实战系列图书

![《LangGraph实战》《LangChain实战》照片](./static/book-photo.jpg)
![《LangGraph实战》购书海报](./static/langgraph-poster.jpg)
![《LangChain实战》购书海报](./static/langchain-poster.jpg)
