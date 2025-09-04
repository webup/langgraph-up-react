"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph_sdk import get_client

# Default test model - use SiliconFlow to avoid API quota issues
TEST_MODEL = "siliconflow:Qwen/Qwen3-8B"


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file for all tests."""
    # Find the project root (where .env is located)
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)

    # Note: Individual tests will check for their specific required keys
    # and skip appropriately. We don't globally skip all tests here.


@pytest.fixture
async def langgraph_client():
    """Create a LangGraph client for e2e testing."""
    return get_client(url="http://127.0.0.1:2024")


@pytest.fixture
async def assistant_id(langgraph_client):
    """Create an assistant with SiliconFlow Qwen3-8B model for testing."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={
            "model": TEST_MODEL,
        },
    )
    assistant_id = assistant["assistant_id"]

    yield assistant_id

    # Cleanup
    try:
        await langgraph_client.assistants.delete(assistant_id)
    except Exception:
        pass  # Ignore cleanup errors


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
