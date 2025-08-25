"""Test suite for tools functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.tools import get_tools, web_search
from tests.test_data import TestModels


class TestGetTools:
    """Test the get_tools function for context-based tool loading."""

    @pytest.mark.asyncio
    async def test_get_tools_with_deepwiki_disabled(self) -> None:
        """Test get_tools returns only web_search when deepwiki is disabled."""
        mock_runtime = MagicMock()
        mock_runtime.context.enable_deepwiki = False

        with patch("common.tools.get_runtime", return_value=mock_runtime):
            tools = await get_tools()

        assert len(tools) == 1
        assert tools[0] == web_search

    @pytest.mark.asyncio
    async def test_get_tools_with_deepwiki_enabled(self) -> None:
        """Test get_tools includes deepwiki tools when enabled."""
        mock_runtime = MagicMock()
        mock_runtime.context.enable_deepwiki = True

        mock_deepwiki_tool1 = AsyncMock()
        mock_deepwiki_tool2 = AsyncMock()
        mock_deepwiki_tools = [mock_deepwiki_tool1, mock_deepwiki_tool2]

        with (
            patch("common.tools.get_runtime", return_value=mock_runtime),
            patch(
                "common.tools.get_deepwiki_tools", return_value=mock_deepwiki_tools
            ) as mock_get_deepwiki,
        ):
            tools = await get_tools()

        assert len(tools) == 3
        assert tools[0] == web_search
        assert tools[1] == mock_deepwiki_tool1
        assert tools[2] == mock_deepwiki_tool2
        mock_get_deepwiki.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tools_with_empty_deepwiki_tools(self) -> None:
        """Test get_tools handles empty deepwiki tools list."""
        mock_runtime = MagicMock()
        mock_runtime.context.enable_deepwiki = True

        with (
            patch("common.tools.get_runtime", return_value=mock_runtime),
            patch(
                "common.tools.get_deepwiki_tools", return_value=[]
            ) as mock_get_deepwiki,
        ):
            tools = await get_tools()

        assert len(tools) == 1
        assert tools[0] == web_search
        mock_get_deepwiki.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_function(self) -> None:
        """Test the web_search function uses runtime context correctly."""
        mock_runtime = MagicMock()
        mock_runtime.context.max_search_results = 10

        mock_tavily = MagicMock()
        mock_tavily.ainvoke = AsyncMock(return_value={"results": ["test result"]})

        with (
            patch("common.tools.get_runtime", return_value=mock_runtime),
            patch(
                "common.tools.TavilySearch", return_value=mock_tavily
            ) as mock_tavily_class,
        ):
            result = await web_search("test query")

        mock_tavily_class.assert_called_once_with(max_results=10)
        mock_tavily.ainvoke.assert_called_once_with({"query": "test query"})
        assert result == {"results": ["test result"]}


class TestToolsIntegration:
    """Integration tests for tools with context system."""

    @pytest.mark.asyncio
    async def test_tools_respect_context_configuration(self) -> None:
        """Test that tools loading respects different context configurations."""
        # This test verifies the integration between Context and get_tools
        from common.context import Context

        # Mock the runtime system to use our test context
        test_context_disabled = Context(
            enable_deepwiki=False, model=TestModels.QWEN_PLUS
        )
        test_context_enabled = Context(enable_deepwiki=True, model=TestModels.QWEN_PLUS)

        mock_runtime_disabled = MagicMock()
        mock_runtime_disabled.context = test_context_disabled

        mock_runtime_enabled = MagicMock()
        mock_runtime_enabled.context = test_context_enabled

        # Test with deepwiki disabled
        with patch("common.tools.get_runtime", return_value=mock_runtime_disabled):
            tools_disabled = await get_tools()

        # Test with deepwiki enabled (mock the actual MCP call)
        with (
            patch("common.tools.get_runtime", return_value=mock_runtime_enabled),
            patch("common.tools.get_deepwiki_tools", return_value=[AsyncMock()]),
        ):
            tools_enabled = await get_tools()

        # Verify different tool counts based on configuration
        assert len(tools_disabled) == 1  # Only web_search
        assert len(tools_enabled) == 2  # web_search + 1 deepwiki tool
