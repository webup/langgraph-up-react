"""Test suite for DeepWiki MCP tools."""

from common.context import Context
from common.mcp import MCP_SERVERS, clear_mcp_cache
from tests.test_data import TestModels


class TestMCPClientConfiguration:
    """Test MCP client configuration and setup."""

    def test_mcp_servers_configuration(self) -> None:
        """Test that MCP servers are properly configured."""
        assert "deepwiki" in MCP_SERVERS
        assert MCP_SERVERS["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"
        assert MCP_SERVERS["deepwiki"]["transport"] == "streamable_http"

    def test_clear_mcp_cache(self) -> None:
        """Test that MCP cache can be cleared."""
        # This should not raise any exceptions
        clear_mcp_cache()


class TestContextIntegration:
    """Test integration with Context system."""

    def test_context_deepwiki_field(self) -> None:
        """Test that Context has enable_deepwiki field."""
        context = Context(enable_deepwiki=False, model=TestModels.QWEN_PLUS)
        assert hasattr(context, "enable_deepwiki")
        assert context.enable_deepwiki is False

        context_enabled = Context(enable_deepwiki=True, model=TestModels.QWEN_PLUS)
        assert context_enabled.enable_deepwiki is True
