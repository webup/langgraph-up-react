"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_sdk import get_client

from tests.test_data import TestModels, TestUrls


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file for all tests."""
    # Find the project root (where .env is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)

    # Ensure required environment variables are available for tests
    # You can add fallback values or skip tests if keys are missing
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        pytest.skip(f"Missing required environment variables: {missing_keys}")


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    mock = Mock()
    mock.bind_tools.return_value = mock
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def sample_ai_response():
    """Create a sample AI response for testing."""
    return AIMessage(
        content="This is a test response from the AI model.", tool_calls=[]
    )


@pytest.fixture
def sample_human_message():
    """Create a sample human message for testing."""
    return HumanMessage(content="This is a test message from human.")


@pytest.fixture
def test_context():
    """Create a test context with common configuration."""
    from common.context import Context

    return Context(
        model=TestModels.QWEN_PLUS,
        system_prompt="You are a helpful AI assistant for testing.",
    )


@pytest.fixture
async def langgraph_client():
    """Create a LangGraph client for e2e testing."""
    return get_client(url=TestUrls.LANGGRAPH_LOCAL)


@pytest.fixture
async def assistant_id(langgraph_client):
    """Get the first available assistant ID for testing."""
    assistants = await langgraph_client.assistants.search()
    if not assistants:
        pytest.skip("No assistants found for e2e testing")
    return assistants[0]["assistant_id"]


@pytest.fixture
async def test_thread(langgraph_client):
    """Create a test thread for e2e testing."""
    thread = await langgraph_client.threads.create()
    yield thread
    # Cleanup could be added here if needed


class TestHelpers:
    """Helper methods for common test operations."""

    @staticmethod
    def assert_valid_response(
        messages: list, expected_content: str | None = None, min_messages: int = 2
    ):
        """Assert that a response has valid structure and content."""
        assert isinstance(messages, list), "Messages should be a list"
        assert len(messages) >= min_messages, (
            f"Should have at least {min_messages} messages"
        )

        # Check final message structure
        final_message = messages[-1]
        assert isinstance(final_message, dict) or hasattr(final_message, "content"), (
            "Final message should have content attribute"
        )

        if expected_content:
            content = str(
                getattr(final_message, "content", final_message.get("content", ""))
            ).lower()
            assert expected_content.lower() in content, (
                f"Expected '{expected_content}' in response content: {content[:200]}..."
            )

    @staticmethod
    def assert_tool_usage(messages: list, tool_name: str = "web_search"):
        """Assert that a specific tool was used in the conversation."""
        tool_found = False
        for msg in messages:
            msg_dict = msg if isinstance(msg, dict) else msg.__dict__

            # Check for tool calls
            if msg_dict.get("tool_calls"):
                for call in msg_dict["tool_calls"]:
                    if isinstance(call, dict) and call.get("name") == tool_name:
                        tool_found = True
                        break

            # Check for tool messages
            if msg_dict.get("type") == "tool" and tool_name in str(
                msg_dict.get("name", "")
            ):
                tool_found = True
                break

        assert tool_found, f"Tool '{tool_name}' should have been used in conversation"

    @staticmethod
    def create_input_state(content: str):
        """Create an InputState for testing."""
        from react_agent.state import InputState

        return InputState(messages=[HumanMessage(content=content)])


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers
