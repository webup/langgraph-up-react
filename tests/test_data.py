"""Centralized test data and constants for the test suite."""

from typing import Any, Dict, List


class TestModels:
    """Test model configurations."""

    QWEN_PLUS = "qwen:qwen-plus"
    QWEN_TURBO = "qwen:qwen-turbo"
    QWQ_32B = "qwen:qwq-32b-preview"
    QVQ_72B = "qwen:qvq-72b-preview"
    OPENAI_GPT4O_MINI = "openai:gpt-4o-mini"
    ANTHROPIC_SONNET = "anthropic:claude-3-sonnet"


class TestQuestions:
    """Common test questions and expected response patterns."""

    SIMPLE_MATH = {
        "question": "What is 2 + 2?",
        "expected_answer": "4",
        "requires_tools": False,
    }

    LANGCHAIN_FOUNDER = {
        "question": "Who is the founder of LangChain?",
        "expected_answer": "harrison",
        "requires_tools": True,
    }

    FRAMEWORK_COMPARISON = {
        "question": "I need to research Python web frameworks. Can you search for information about FastAPI and Django, then compare them and recommend which one would be better for a beginner to learn?",
        "expected_keywords": ["fastapi", "django", "compare", "beginner", "recommend"],
        "requires_tools": True,
    }


class TestApiKeys:
    """Test API key configurations."""

    MOCK_OPENAI = "test-openai-key"
    MOCK_DASHSCOPE = "test-key"
    MOCK_TAVILY = "test-tavily-key"


class TestUrls:
    """Test URL configurations."""

    LANGGRAPH_LOCAL = "http://localhost:2024"
    DASHSCOPE_PRC = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class TestDataBuilder:
    """Builder for creating test data structures."""

    @staticmethod
    def create_message_input(content: str, role: str = "human") -> Dict[str, Any]:
        """Create a standard message input structure."""
        return {"messages": [{"role": role, "content": content}]}

    @staticmethod
    def create_mock_ai_response(
        content: str, tool_calls: List[Dict[str, Any]] | None = None
    ) -> Dict[str, Any]:
        """Create a mock AI response structure."""
        return {"content": content, "tool_calls": tool_calls or [], "type": "ai"}

    @staticmethod
    def create_tool_call(
        name: str, args: Dict[str, Any], call_id: str = "call_123"
    ) -> Dict[str, Any]:
        """Create a tool call structure."""
        return {"name": name, "args": args, "id": call_id}


# Common assertion patterns
class TestAssertions:
    """Common assertion patterns for test validation."""

    @staticmethod
    def validate_message_structure(messages: List[Dict]) -> bool:
        """Validate that messages have proper structure."""
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "content" not in msg or "type" not in msg:
                return False

        return True

    @staticmethod
    def validate_tool_usage(
        messages: List[Dict], tool_name: str = "web_search"
    ) -> bool:
        """Validate that specified tool was used in conversation."""
        for msg in messages:
            if isinstance(msg, dict):
                # Check for tool calls
                if msg.get("tool_calls"):
                    for call in msg["tool_calls"]:
                        if isinstance(call, dict) and call.get("name") == tool_name:
                            return True
                # Check for tool messages
                if msg.get("type") == "tool" and tool_name in str(msg.get("name", "")):
                    return True
        return False
