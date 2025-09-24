# Repository Guidelines

## Project Structure & Module Organization
- Core agent logic lives in `src/react_agent/` (`graph.py` orchestrates the ReAct loop); shared utilities reside in `src/common/`.
- Shared Pydantic schemas should inherit from `src/common/basemodel.AgentBaseModel`; keep the base class in that module so `src/common/models/` stays focused on provider integrations.
- Packaging metadata and templates are exported via `langgraph/templates/react_agent`; add new modules under `src/` and expose them through `__init__.py` as needed.
- Tests are split into `tests/unit_tests/`, `tests/integration_tests/`, and `tests/e2e_tests/`; evaluation harnesses live in `tests/evaluations/` with scenario scripts.
- Static assets such as screenshots or fixtures belong in `static/` or the relevant `tests/cassettes/` folder.

## Build, Test, and Development Commands
- Install all deps with `uv sync --dev`; activate env through `uv run ...` or the generated `.venv`.
- Launch the local graph runtime with `make dev` (headless) or `make dev_ui` (opens LangGraph Studio).
- Run targeted suites via `make test_unit`, `make test_integration`, `make test_e2e`, or `make test_all`; watch mode is available through `make test_watch_*` targets.
- Execute evaluation scenarios through `make eval_graph`, `make eval_multiturn`, or persona/model-specific targets like `make eval_graph_qwen`.

## Coding Style & Naming Conventions
- Python 3.11+ with 4-space indentation; favor type annotations and Google-style docstrings (enforced by Ruff + pydocstyle).
- Run `make lint` before committing; it invokes Ruff formatting/isort checks and strict MyPy over `src/`.
- Use snake_case for modules and functions, PascalCase for classes, and uppercase ENV names; keep public tool IDs descriptive (e.g., `search_tool`).

## Testing Guidelines
- Write tests with `pytest`; mirror implementation folders (e.g., `tests/unit_tests/common/` for `src/common/`).
- Prefer deterministic fixtures and reuse cassette data in `tests/cassettes/` for external calls.
- For new behaviors add unit coverage first, then integration/e2e if the agent flow changes; verify locally with the nearest `make test_*` target.

## Commit & Pull Request Guidelines
- Follow the existing conventional-emoji prefix style (`📚 docs:`, `♻️ refactor:`); keep the summary imperative and under 72 chars.
- Reference issues in the body (`Fixes #123`) and note evaluation/test results.
- PRs should explain the change, list verification commands, and attach screenshots or trace links when UI or agent output changes.

## Security & Configuration Tips
- Keep secrets in `.env` only; never commit API keys. Update `.env.example` when adding new configuration knobs.
- Document any external service requirements (SiliconFlow, Tavily, OpenAI) and ensure fallback behaviors when keys are absent.
