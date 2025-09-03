"""Integration tests for graph workflow with new model providers."""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from common.context import Context
from react_agent.graph import graph
from react_agent.state import InputState


class TestModelIntegrationWorkflow:
    """Integration tests for graph workflow with different model providers."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=False)
    async def test_qwen_model_workflow(self):
        """Test complete workflow with Qwen model provider."""
        with patch("common.models.qwen.ChatQwen") as mock_chat_qwen:
            # Mock the Qwen model
            mock_model = Mock()
            mock_chat_qwen.return_value = mock_model

            # Mock bind_tools to return the model itself
            mock_model.bind_tools.return_value = mock_model

            # Mock ainvoke to return a simple response without tool calls
            mock_response = AIMessage(
                content="Hello! I understand you want to test the workflow.",
                tool_calls=[],
            )
            mock_model.ainvoke = AsyncMock(return_value=mock_response)

            # Create context with Qwen model
            context = Context(model="qwen:qwen-plus")
            input_state = InputState(
                messages=[HumanMessage(content="Hello, test the workflow")]
            )

            # Run the graph
            result = await graph.ainvoke(input_state, context=context)

            # Verify the workflow completed
            assert len(result["messages"]) == 2  # Original + response
            assert isinstance(result["messages"][-1], AIMessage)
            assert "Hello!" in result["messages"][-1].content

            # Verify Qwen model was used
            mock_chat_qwen.assert_called_once_with(
                model="qwen-plus",
                api_key="test-key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    @pytest.mark.asyncio
    async def test_openai_model_workflow(self):
        """Test complete workflow with OpenAI model provider."""
        with patch("common.utils.init_chat_model") as mock_init:
            # Mock the OpenAI model
            mock_model = Mock()
            mock_init.return_value = mock_model

            # Mock bind_tools to return the model itself
            mock_model.bind_tools.return_value = mock_model

            # Mock ainvoke to return a simple response without tool calls
            mock_response = AIMessage(
                content="This is a test response from OpenAI model.", tool_calls=[]
            )
            mock_model.ainvoke = AsyncMock(return_value=mock_response)

            # Create context with OpenAI model
            context = Context(model="openai:gpt-4o-mini")
            input_state = InputState(
                messages=[HumanMessage(content="Test OpenAI workflow")]
            )

            # Run the graph
            result = await graph.ainvoke(input_state, context=context)

            # Verify the workflow completed
            assert len(result["messages"]) == 2
            assert isinstance(result["messages"][-1], AIMessage)
            assert "OpenAI model" in result["messages"][-1].content

            # Verify OpenAI model was used
            mock_init.assert_called_once_with("gpt-4o-mini", model_provider="openai")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=False)
    async def test_qwen_model_with_tools_workflow(self):
        """Test workflow with Qwen model that uses tools."""
        with (
            patch("common.models.qwen.ChatQwen") as mock_chat_qwen,
            patch("common.tools.get_tools") as mock_get_tools,
        ):
            # Mock the Qwen model
            mock_model = Mock()
            mock_chat_qwen.return_value = mock_model
            mock_model.bind_tools.return_value = mock_model

            # Mock tool call response first, then final response
            tool_call_response = AIMessage(
                content="I'll search for that information.",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "test query"},
                        "id": "call_123",
                    }
                ],
            )

            final_response = AIMessage(
                content="Based on the search results, here's what I found.",
                tool_calls=[],
            )

            mock_model.ainvoke = AsyncMock(
                side_effect=[tool_call_response, final_response]
            )

            # Create a proper async function mock for the tool
            async def mock_web_search(query: str):
                return {"results": [{"content": "Test search result"}]}

            # Mock get_tools to return our mock function
            mock_get_tools.return_value = [mock_web_search]

            # Create context and input
            context = Context(model="qwen:qwen-plus")
            input_state = InputState(
                messages=[HumanMessage(content="Search for Python tutorials")]
            )

            # Run the graph
            result = await graph.ainvoke(input_state, context=context)

            # Verify workflow completed with tool usage
            assert (
                len(result["messages"]) >= 3
            )  # Input + tool call + tool result + final response
            assert isinstance(result["messages"][-1], AIMessage)

            # Verify model was called multiple times (tool call + final response)
            assert mock_model.ainvoke.call_count == 2

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=False)
    async def test_qwq_model_workflow(self):
        """Test complete workflow with QwQ model provider uses ChatQwQ."""
        with patch("common.models.qwen.ChatQwQ") as mock_chat_qwq:
            # Mock the QwQ model
            mock_model = Mock()
            mock_chat_qwq.return_value = mock_model

            # Mock bind_tools to return the model itself
            mock_model.bind_tools.return_value = mock_model

            # Mock ainvoke to return a simple response without tool calls
            mock_response = AIMessage(
                content="I need to think step by step about this problem.",
                tool_calls=[],
            )
            mock_model.ainvoke = AsyncMock(return_value=mock_response)

            # Create context with QwQ model
            context = Context(model="qwen:qwq-32b-preview")
            input_state = InputState(
                messages=[HumanMessage(content="Solve this math problem: 2+2")]
            )

            # Run the graph
            result = await graph.ainvoke(input_state, context=context)

            # Verify the workflow completed
            assert len(result["messages"]) == 2  # Original + response
            assert isinstance(result["messages"][-1], AIMessage)
            assert "think step by step" in result["messages"][-1].content

            # Verify QwQ model (ChatQwQ) was used
            mock_chat_qwq.assert_called_once_with(
                model="qwq-32b-preview",
                api_key="test-key",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def test_model_separator_format(self):
        """Test that colon separator is properly supported."""
        test_cases = [
            "qwen:qwen-plus",
            "qwen:qwen-turbo",
            "qwen:qwq-32b-preview",
            "openai:gpt-4o-mini",
            "anthropic:claude-3-sonnet",
        ]

        for model_spec in test_cases:
            context = Context(model=model_spec)
            assert context.model == model_spec
            assert ":" in context.model  # Verify colon separator format

    def test_invalid_model_format_raises_error(self):
        """Test that invalid model formats raise appropriate errors."""
        # Import the utils function directly to test it
        from common.utils import load_chat_model

        with pytest.raises(ValueError):
            load_chat_model("invalid-format-no-separator")
