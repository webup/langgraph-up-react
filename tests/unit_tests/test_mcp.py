"""Comprehensive unit tests for the MCP module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.mcp import (
    MCP_SERVERS,
    add_mcp_server,
    clear_mcp_cache,
    get_all_mcp_tools,
    get_deepwiki_tools,
    get_mcp_client,
    get_mcp_tools,
)


class TestMCPClientInitialization:
    """Test MCP client initialization and management."""

    @pytest.mark.asyncio
    async def test_get_mcp_client_initialization(self) -> None:
        """Test MCP client is initialized with default servers."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.MultiServerMCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = await get_mcp_client()

            assert client is mock_client
            mock_client_class.assert_called_once_with(MCP_SERVERS)

    @pytest.mark.asyncio
    async def test_get_mcp_client_with_custom_configs(self) -> None:
        """Test MCP client initialization with custom server configurations."""
        clear_mcp_cache()  # Ensure clean state

        custom_configs = {
            "test_server": {
                "url": "https://test.example.com/mcp",
                "transport": "streamable_http",
            }
        }

        with patch("common.mcp.MultiServerMCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = await get_mcp_client(custom_configs)

            assert client is mock_client
            mock_client_class.assert_called_once_with(custom_configs)

    @pytest.mark.asyncio
    async def test_get_mcp_client_singleton_behavior(self) -> None:
        """Test that MCP client follows singleton pattern."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.MultiServerMCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client1 = await get_mcp_client()
            client2 = await get_mcp_client()

            assert client1 is client2
            # Should only be called once due to singleton pattern
            mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mcp_client_initialization_failure(self) -> None:
        """Test MCP client handles initialization failures gracefully."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.MultiServerMCPClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            client = await get_mcp_client()

            assert client is None


class TestMCPToolsLoading:
    """Test MCP tools loading and caching functionality."""

    @pytest.mark.asyncio
    async def test_get_mcp_tools_successful_loading(self) -> None:
        """Test successful MCP tools loading from a server."""
        clear_mcp_cache()  # Ensure clean state

        mock_tool1 = AsyncMock()
        mock_tool2 = AsyncMock()
        mock_tools = [mock_tool1, mock_tool2]

        with patch("common.mcp.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_get_client.return_value = mock_client

            tools = await get_mcp_tools("test_server")

            assert len(tools) == 2
            assert tools[0] is mock_tool1
            assert tools[1] is mock_tool2
            mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mcp_tools_caching_behavior(self) -> None:
        """Test that MCP tools are cached after first load."""
        clear_mcp_cache()  # Ensure clean state

        mock_tools = [AsyncMock(), AsyncMock()]

        with patch("common.mcp.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=mock_tools)
            mock_get_client.return_value = mock_client

            # First call should load from client
            tools1 = await get_mcp_tools("test_server")
            # Second call should use cache
            tools2 = await get_mcp_tools("test_server")

            assert tools1 == tools2
            # Client should only be called once due to caching
            mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mcp_tools_client_unavailable(self) -> None:
        """Test MCP tools loading when client is unavailable."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.get_mcp_client") as mock_get_client:
            mock_get_client.return_value = None

            tools = await get_mcp_tools("test_server")

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_mcp_tools_loading_failure(self) -> None:
        """Test MCP tools loading handles failures gracefully."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=Exception("Network error"))
            mock_get_client.return_value = mock_client

            tools = await get_mcp_tools("test_server")

            assert tools == []

    @pytest.mark.asyncio
    async def test_get_deepwiki_tools(self) -> None:
        """Test DeepWiki-specific tools loading."""
        clear_mcp_cache()  # Ensure clean state

        mock_tools = [AsyncMock(), AsyncMock()]

        with patch("common.mcp.get_mcp_tools") as mock_get_mcp_tools:
            mock_get_mcp_tools.return_value = mock_tools

            tools = await get_deepwiki_tools()

            assert tools == mock_tools
            mock_get_mcp_tools.assert_called_once_with("deepwiki")

    @pytest.mark.asyncio
    async def test_get_all_mcp_tools(self) -> None:
        """Test loading all tools from all configured servers."""
        clear_mcp_cache()  # Ensure clean state

        # Mock tools from different servers
        deepwiki_tools = [AsyncMock(), AsyncMock()]

        with patch("common.mcp.get_mcp_tools") as mock_get_mcp_tools:
            mock_get_mcp_tools.return_value = deepwiki_tools

            all_tools = await get_all_mcp_tools()

            # Should include tools from all servers
            assert len(all_tools) == 2
            assert all_tools == deepwiki_tools
            # Should be called for each server in MCP_SERVERS
            mock_get_mcp_tools.assert_called_with("deepwiki")


class TestMCPServerManagement:
    """Test MCP server configuration management."""

    def test_add_mcp_server(self) -> None:
        """Test adding a new MCP server configuration."""
        original_servers = dict(MCP_SERVERS)  # Backup original

        try:
            new_config = {
                "url": "https://new.example.com/mcp",
                "transport": "streamable_http",
            }

            add_mcp_server("new_server", new_config)

            assert "new_server" in MCP_SERVERS
            assert MCP_SERVERS["new_server"] == new_config

        finally:
            # Restore original configuration
            MCP_SERVERS.clear()
            MCP_SERVERS.update(original_servers)
            clear_mcp_cache()

    def test_clear_mcp_cache(self) -> None:
        """Test that MCP cache clearing works properly."""
        # This should not raise any exceptions
        clear_mcp_cache()

        # Verify that subsequent calls work (cache is properly cleared)
        clear_mcp_cache()


class TestMCPServerConfiguration:
    """Test MCP server configuration validation."""

    def test_mcp_servers_default_configuration(self) -> None:
        """Test that default MCP server configurations are valid."""
        assert isinstance(MCP_SERVERS, dict)
        assert "deepwiki" in MCP_SERVERS

        deepwiki_config = MCP_SERVERS["deepwiki"]
        assert "url" in deepwiki_config
        assert "transport" in deepwiki_config
        assert deepwiki_config["url"] == "https://mcp.deepwiki.com/mcp"
        assert deepwiki_config["transport"] == "streamable_http"

    def test_mcp_servers_structure(self) -> None:
        """Test that MCP server configurations have the expected structure."""
        for server_name, config in MCP_SERVERS.items():
            assert isinstance(server_name, str)
            assert isinstance(config, dict)
            assert "url" in config
            assert "transport" in config
            assert isinstance(config["url"], str)
            assert isinstance(config["transport"], str)


class TestMCPErrorHandling:
    """Test error handling scenarios in MCP functionality."""

    @pytest.mark.asyncio
    async def test_concurrent_mcp_client_access(self) -> None:
        """Test concurrent access to MCP client doesn't cause issues."""
        clear_mcp_cache()  # Ensure clean state

        with patch("common.mcp.MultiServerMCPClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Simulate concurrent access
            import asyncio

            clients = await asyncio.gather(
                get_mcp_client(),
                get_mcp_client(),
                get_mcp_client(),
            )

            # All should return the same client instance
            assert all(client is mock_client for client in clients)
            # Should only initialize once
            mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_tools_different_servers_independent_caching(self) -> None:
        """Test that different servers have independent tool caches."""
        clear_mcp_cache()  # Ensure clean state

        server1_tools = [AsyncMock()]
        server2_tools = [AsyncMock(), AsyncMock()]

        def mock_get_tools_side_effect():
            # Different tools for different calls
            call_count = mock_client.get_tools.call_count
            if call_count == 1:
                return server1_tools
            elif call_count == 2:
                return server2_tools
            return []

        with patch("common.mcp.get_mcp_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(side_effect=mock_get_tools_side_effect)
            mock_get_client.return_value = mock_client

            tools1 = await get_mcp_tools("server1")
            tools2 = await get_mcp_tools("server2")

            assert len(tools1) == 1
            assert len(tools2) == 2
            assert tools1 != tools2

            # Verify caching works independently
            cached_tools1 = await get_mcp_tools("server1")
            cached_tools2 = await get_mcp_tools("server2")

            assert tools1 == cached_tools1
            assert tools2 == cached_tools2
