# Roadmap

This document outlines the planned upgrades and enhancements for the LangGraph ReAct Agent template.

## v0.3.0 - Prototyping & Deployment Infrastructure (Planning)

> Reference: [LangGraph Platform Standalone Server Deployment](https://docs.langchain.com/langgraph-platform/deploy-standalone-server#how-to-deploy-self-hosted-standalone-server)

### LangGraph Platform Standalone Server (Prototyping)
- **Template Integration**: Pre-configured deployment files directly in template for immediate developer use
- **Self-Hosted Setup**: Complete standalone server configuration following official LangGraph Platform guidelines
- **Container Setup**: Docker builds optimized for LangGraph Platform standalone server deployment

### Deployment Configurations in Template
- **Docker Compose**: Ready-to-use `docker-compose.yml` for local and prototype standalone server deployments
- **Kubernetes Manifests**: K8s deployment configurations with ConfigMaps, Secrets, and service definitions for prototype environments
- **Development Workflow**: Seamless transition from local development (`make dev`) to containerized prototyping

## v0.2.0 - Multi-Provider Evaluation Framework (Planning)

### SiliconFlow Integration
- **Provider Support**: Add standalone SiliconFlow provider prefix in `load_chat_model`
- **Regional Coverage**: Support both PRC and international API bases for SiliconFlow
- **SOTA OSS Models**: Leverage state-of-the-art open-source models for comprehensive evaluation

### Agent Evaluation
- **OpenEvals Integration**: Implement basic evaluation framework using OpenEvals
- **AgentEvals Setup**: Add AgentEvals for comprehensive agent performance testing
- **Evaluation Tests**: Create structured evaluation test suites to measure agent effectiveness across multiple model providers

## v0.1.0 - Core Infrastructure Enhancements (Completed)

### Dynamic Tool Integration
- **DeepWiki MCP Integration**: Added dynamic loading of DeepWiki MCP tools for open source documentation
- **Tool Discovery**: Implemented runtime tool loading with graceful fallback when services unavailable
- **Configuration Toggle**: Added `enable_deepwiki` flag for optional documentation tool integration

### Model Provider Expansion  
- **Qwen Integration**: Full support for Qwen models via `langchain-qwq` with QwQ/QvQ reasoning models
- **Regional API Support**: Dynamic base URL selection for PRC/international endpoints via `REGION` env var
- **Multi-Provider Architecture**: Unified model loading supporting OpenAI-compatible and Qwen-specific providers

### Code Architecture Refactoring
- **Common Module**: Extracted shared components (context, models, tools, utils, prompts) into reusable `common/` module
- **Modular Agent Structure**: Prepared scalable architecture supporting multiple agents under `src/`
- **Tool Naming**: Renamed `search` to `web_search` for API compatibility

### Comprehensive Test Suite
- **60+ Test Cases**: Complete coverage across unit/integration/e2e categories with structured test data
- **ReAct Pattern Validation**: E2E tests verifying tool-model interaction loops
- **Error Handling Coverage**: Extensive edge case and failure mode testing

### Development Tools
- **Development Server**: `make dev` and `make dev_ui` commands for LangGraph development
- **Test Organization**: Structured test commands with watch modes for unit/integration/e2e testing
- **Async Test Support**: Proper `pytest-asyncio` configuration

---

Each version builds upon the previous foundation, progressively enhancing the agent's capabilities, developer experience, and testing infrastructure.