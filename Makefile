.PHONY: all format lint test test_unit test_integration test_e2e test_all evals eval_graph eval_multiturn eval_graph_qwen eval_graph_glm eval_multiturn_polite eval_multiturn_hacker test_watch test_watch_unit test_watch_integration test_watch_e2e test_profile extended_tests dev dev_ui

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING
######################

# Legacy test command (defaults to unit and integration tests for backward compatibility)
test: test_unit test_integration

# Specific test targets
test_unit:
	uv run python -m pytest tests/unit_tests/

test_integration:
	uv run python -m pytest tests/integration_tests/

test_e2e:
	uv run python -m pytest tests/e2e_tests/

test_all:
	uv run python -m pytest tests/

######################
# EVALUATIONS
######################

# Comprehensive evaluation suite
evals: eval_graph eval_multiturn

# Graph trajectory evaluation (scenario-specific LLM-as-judge)
eval_graph:
	cd tests/evaluations && python graph.py --verbose

# Multi-turn chat evaluation (role-persona simulations)
eval_multiturn:
	cd tests/evaluations && python multiturn.py --verbose

# Run specific evaluation scenarios
eval_graph_qwen:
	cd tests/evaluations && python graph.py --model siliconflow:Qwen/Qwen3-8B --verbose

eval_graph_glm:
	cd tests/evaluations && python graph.py --model siliconflow:THUDM/GLM-4-9B-0414 --verbose

eval_multiturn_polite:
	cd tests/evaluations && python multiturn.py --persona polite --verbose

eval_multiturn_hacker:
	cd tests/evaluations && python multiturn.py --persona hacker --verbose

######################
# WATCH MODES
######################

# Watch mode for tests
test_watch: test_watch_unit

test_watch_unit:
	uv run python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_watch_integration:
	uv run python -m ptw --snapshot-update --now . -- -vv tests/integration_tests

test_watch_e2e:
	uv run python -m ptw --snapshot-update --now . -- -vv tests/e2e_tests

test_profile:
	uv run python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	uv run python -m pytest --only-extended tests/unit_tests/

######################
# DEVELOPMENT
######################

dev:
	uv run langgraph dev --no-browser

dev_ui:
	uv run langgraph dev


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint:
	uv run python -m ruff check .
	uv run python -m ruff format src --diff
	uv run python -m ruff check --select I src
	uv run python -m mypy --strict src
	mkdir -p .mypy_cache && uv run python -m mypy --strict src --cache-dir .mypy_cache

lint_diff lint_package:
	uv run python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_tests:
	uv run python -m ruff check tests --fix
	uv run python -m ruff format tests
	# Skip mypy for tests to allow more flexible typing

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run codespell --toml pyproject.toml

spell_fix:
	uv run codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'DEVELOPMENT:'
	@echo 'dev                          - run langgraph dev without browser'
	@echo 'dev_ui                       - run langgraph dev with browser'
	@echo ''
	@echo 'TESTING:'
	@echo 'test                         - run unit tests (default)'
	@echo 'test_unit                    - run unit tests only'
	@echo 'test_integration             - run integration tests only'
	@echo 'test_e2e                     - run e2e tests only'
	@echo 'test_all                     - run all tests (unit + integration + e2e)'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_watch_unit              - run unit tests in watch mode'
	@echo 'test_watch_integration       - run integration tests in watch mode'
	@echo 'test_watch_e2e               - run e2e tests in watch mode'
	@echo ''
	@echo 'EVALUATIONS:'
	@echo 'evals                        - run comprehensive evaluation suite (all models)'
	@echo 'eval_graph                   - run graph trajectory evaluations (LLM-as-judge)'
	@echo 'eval_multiturn               - run multi-turn chat evaluations (role-persona)'
	@echo 'eval_graph_qwen              - run graph evaluation with Qwen/Qwen3-8B model'
	@echo 'eval_graph_glm               - run graph evaluation with THUDM/GLM-4-9B model'
	@echo 'eval_multiturn_polite        - run multiturn with polite persona only'
	@echo 'eval_multiturn_hacker        - run multiturn with hacker persona only'
	@echo ''
	@echo 'CODE QUALITY:'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters (ruff + mypy on src/)'
	@echo 'lint_tests                   - run linters on tests (ruff only, no mypy)'
	@echo 'lint_package                 - run linters on src/ only'

