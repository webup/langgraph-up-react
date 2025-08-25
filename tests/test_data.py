"""Centralized test data and constants for the test suite."""


class TestModels:
    """Test model configurations."""

    QWEN_PLUS = "qwen:qwen-plus"
    QWEN_TURBO = "qwen:qwen-turbo"
    QWQ_32B = "qwen:qwq-32b-preview"
    QVQ_72B = "qwen:qvq-72b-preview"
    OPENAI_GPT4O_MINI = "openai:gpt-4o-mini"
    ANTHROPIC_SONNET = "anthropic:claude-4-sonnet"


class TestQuestions:
    """Common test questions and expected response patterns."""

    SIMPLE_MATH = {
        "question": "What is 2 + 2?",
        "expected_answer": "4",
        "requires_tools": False,
    }


class TestApiKeys:
    """Test API key configurations."""

    MOCK_DASHSCOPE = "test-key"
