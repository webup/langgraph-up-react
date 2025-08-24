.PHONY: all format lint test test_unit test_integration test_e2e test_all test_watch test_watch_unit test_watch_integration test_watch_e2e test_profile extended_tests dev dev_ui

# Default target executed when no arguments are given to make.
all: help

######################
# TESTING
######################

# Legacy test command (defaults to unit tests for backward compatibility)
test:
	python -m pytest tests/unit_tests/

# Specific test targets
test_unit:
	python -m pytest tests/unit_tests/

test_integration:
	python -m pytest tests/integration_tests/

test_e2e:
	python -m pytest tests/e2e_tests/

test_all:
	python -m pytest tests/

# Watch mode for tests
test_watch: test_watch_unit

test_watch_unit:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_watch_integration:
	python -m ptw --snapshot-update --now . -- -vv tests/integration_tests

test_watch_e2e:
	python -m ptw --snapshot-update --now . -- -vv tests/e2e_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended tests/unit_tests/

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
	python -m ruff check .
	python -m ruff format src --diff
	python -m ruff check --select I src
	python -m mypy --strict src
	mkdir -p .mypy_cache && python -m mypy --strict src --cache-dir .mypy_cache

lint_diff lint_package:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_tests:
	python -m ruff check tests --fix
	python -m ruff format tests
	# Skip mypy for tests to allow more flexible typing

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

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
	@echo 'CODE QUALITY:'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters (ruff + mypy on src/)'
	@echo 'lint_tests                   - run linters on tests (ruff only, no mypy)'
	@echo 'lint_package                 - run linters on src/ only'

