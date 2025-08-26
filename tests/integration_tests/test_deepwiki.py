"""Integration tests for DeepWiki functionality."""

import os
from unittest.mock import patch

import pytest

from common.context import Context
from common.mcp import MCP_SERVERS, clear_mcp_cache, get_deepwiki_tools


class TestMCPIntegration:
    """Integration tests for MCP client functionality."""

    def test_mcp_server_configuration_exists(self) -> None:
        """Test that MCP server configuration is properly set up."""
        assert "deepwiki" in MCP_SERVERS
        config = MCP_SERVERS["deepwiki"]
        assert "url" in config
        assert "transport" in config
        assert config["transport"] == "streamable_http"

    def test_mcp_cache_management(self) -> None:
        """Test that MCP cache can be cleared without issues."""
        # This should not raise any exceptions
        clear_mcp_cache()

    @patch.dict(os.environ, {"ENABLE_DEEPWIKI": "false"}, clear=True)
    def test_context_with_deepwiki_disabled(self) -> None:
        """Test context creation with deepwiki explicitly disabled."""
        context = Context(enable_deepwiki=False)
        # Environment variable overrides explicit setting only if explicit value equals default
        # Since we explicitly set enable_deepwiki=False and default is False, no override should occur
        assert not context.enable_deepwiki

    @patch.dict(os.environ, {"ENABLE_DEEPWIKI": "true"}, clear=False)
    def test_context_with_deepwiki_enabled_via_env(self) -> None:
        """Test context creation with deepwiki enabled via environment."""
        context = Context()
        # Environment variable should be converted to boolean when current value equals default
        assert context.enable_deepwiki

    @pytest.mark.asyncio
    async def test_deepwiki_tools_loading_mechanism(self) -> None:
        """Test the deepwiki tools loading mechanism (without actual network calls)."""
        # Clear cache first to ensure clean state
        clear_mcp_cache()

        # This test verifies the loading mechanism exists
        # Actual network calls are tested in live environments only
        assert callable(get_deepwiki_tools)


class TestDeepWikiConfiguration:
    """Test deepwiki configuration and setup."""

    def test_deepwiki_context_field_exists(self) -> None:
        """Test that the enable_deepwiki field exists in Context."""
        context = Context()
        assert hasattr(context, "enable_deepwiki")

    def test_deepwiki_can_be_configured(self) -> None:
        """Test that deepwiki can be explicitly enabled and disabled."""
        context_disabled = Context(enable_deepwiki=False)
        context_enabled = Context(enable_deepwiki=True)

        # These should not be the same
        assert context_disabled.enable_deepwiki != context_enabled.enable_deepwiki
