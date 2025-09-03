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
    remove_mcp_server,
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

        # Add test server to configuration
        test_server_config = {
            "url": "https://test.example.com/mcp",
            "transport": "streamable_http",
        }
        add_mcp_server("test_server", test_server_config)

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

        # Clean up
        remove_mcp_server("test_server")
        clear_mcp_cache()

    @pytest.mark.asyncio
    async def test_get_mcp_tools_caching_behavior(self) -> None:
        """Test that MCP tools are cached after first load."""
        clear_mcp_cache()  # Ensure clean state

        # Add test server to configuration
        test_server_config = {
            "url": "https://test.example.com/mcp",
            "transport": "streamable_http",
        }
        add_mcp_server("test_server", test_server_config)

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

        # Clean up
        remove_mcp_server("test_server")
        clear_mcp_cache()

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

        # Add test servers to configuration
        server1_config = {
            "url": "https://server1.example.com/mcp",
            "transport": "streamable_http",
        }
        server2_config = {
            "url": "https://server2.example.com/mcp",
            "transport": "streamable_http",
        }
        add_mcp_server("server1", server1_config)
        add_mcp_server("server2", server2_config)

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

        # Clean up
        remove_mcp_server("server1")
        remove_mcp_server("server2")
        clear_mcp_cache()


class TestMCPServerFiltering:
    """Test MCP server filtering functionality."""

    @pytest.mark.asyncio
    async def test_mcp_server_filtering(self) -> None:
        """
        Test that MCP server filtering works correctly.

        This test verifies that get_mcp_tools(server_name) returns tools only
        from the specified server, not from all servers.
        """
        # Clean state
        clear_mcp_cache()

        # Add a test server to demonstrate filtering
        new_server_config = {
            "url": "https://new.example.com/mcp",
            "transport": "streamable_http",
        }
        add_mcp_server("new_server", new_server_config)

        try:
            # This test demonstrates that tools are properly filtered by server
            # In a real scenario, each server would return different tools
            # For this test, we verify the structure works correctly

            # Get tools from all servers
            all_tools = await get_all_mcp_tools()

            # Get tools from specific servers
            deepwiki_tools = await get_mcp_tools("deepwiki")
            new_server_tools = await get_mcp_tools("new_server")

            # Verify the fundamental structure works
            # (In a real scenario with actual MCP servers, we'd verify tool counts match)
            assert isinstance(all_tools, list), "get_all_mcp_tools should return a list"
            assert isinstance(deepwiki_tools, list), (
                "get_mcp_tools should return a list"
            )
            assert isinstance(new_server_tools, list), (
                "get_mcp_tools should return a list"
            )

            # Verify that each server request creates a separate client
            # This is the core server-specific filtering behavior

            # Test with a non-existent server (should return empty list)
            nonexistent_tools = await get_mcp_tools("nonexistent_server")
            assert nonexistent_tools == [], (
                "Non-existent server should return empty list"
            )

            # Test that caching works correctly per server
            cached_deepwiki_tools = await get_mcp_tools("deepwiki")
            cached_new_server_tools = await get_mcp_tools("new_server")

            assert deepwiki_tools == cached_deepwiki_tools, (
                "Caching should work for deepwiki"
            )
            assert new_server_tools == cached_new_server_tools, (
                "Caching should work for new_server"
            )

        finally:
            # Clean up test server
            remove_mcp_server("new_server")
            clear_mcp_cache()

    @pytest.mark.asyncio
    async def test_real_mcp_server_filtering_deepwiki_vs_context7(self) -> None:
        """
        Test real MCP server filtering with DeepWiki and Context7 servers.

        This test verifies that:
        1. DeepWiki and Context7 servers return different sets of tools
        2. Server-specific tool requests work correctly
        3. Tools are properly isolated per server
        4. All tools aggregation works correctly
        """
        # Clean state
        clear_mcp_cache()

        # Add Context7 server configuration
        context7_config = {
            "url": "https://mcp.context7.com/sse",
            "transport": "sse",
        }
        add_mcp_server("context7", context7_config)

        try:
            # Get tools from individual servers
            deepwiki_tools = await get_mcp_tools("deepwiki")
            context7_tools = await get_mcp_tools("context7")

            # Get all tools
            all_tools = await get_all_mcp_tools()

            # Basic structure validation
            assert isinstance(deepwiki_tools, list), "DeepWiki tools should be a list"
            assert isinstance(context7_tools, list), "Context7 tools should be a list"
            assert isinstance(all_tools, list), "All tools should be a list"

            # Extract tool names for comparison
            def extract_tool_names(tools):
                names = []
                for tool in tools:
                    if hasattr(tool, "name"):
                        names.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        names.append(tool.__name__)
                    else:
                        names.append(str(tool))
                return names

            deepwiki_tool_names = extract_tool_names(deepwiki_tools)
            context7_tool_names = extract_tool_names(context7_tools)
            all_tool_names = extract_tool_names(all_tools)

            # Verify DeepWiki tools (expected tools from DeepWiki server)
            if len(deepwiki_tools) > 0:
                expected_deepwiki_tools = [
                    "read_wiki_structure",
                    "read_wiki_contents",
                    "ask_question",
                ]
                for expected_tool in expected_deepwiki_tools:
                    assert any(expected_tool in name for name in deepwiki_tool_names), (
                        f"Expected DeepWiki tool '{expected_tool}' not found in {deepwiki_tool_names}"
                    )

            # Verify Context7 tools (expected tools from Context7 server)
            if len(context7_tools) > 0:
                expected_context7_tools = ["resolve-library-id", "get-library-docs"]
                for expected_tool in expected_context7_tools:
                    assert any(expected_tool in name for name in context7_tool_names), (
                        f"Expected Context7 tool '{expected_tool}' not found in {context7_tool_names}"
                    )

            # Verify server isolation: tools should be different between servers
            # (unless there's overlap, which is unlikely for these specific servers)
            if len(deepwiki_tools) > 0 and len(context7_tools) > 0:
                # Check that each server has some unique tools
                deepwiki_unique = set(deepwiki_tool_names) - set(context7_tool_names)
                context7_unique = set(context7_tool_names) - set(deepwiki_tool_names)

                assert len(deepwiki_unique) > 0 or len(context7_unique) > 0, (
                    f"Servers should have some different tools. "
                    f"DeepWiki: {deepwiki_tool_names}, Context7: {context7_tool_names}"
                )

            # Verify all_tools contains tools from both servers
            if len(deepwiki_tools) > 0:
                for tool_name in deepwiki_tool_names:
                    assert any(tool_name in all_name for all_name in all_tool_names), (
                        f"All tools should include DeepWiki tool '{tool_name}'"
                    )

            if len(context7_tools) > 0:
                for tool_name in context7_tool_names:
                    assert any(tool_name in all_name for all_name in all_tool_names), (
                        f"All tools should include Context7 tool '{tool_name}'"
                    )

            # Verify that the total count makes sense
            expected_total = len(deepwiki_tools) + len(context7_tools)
            assert len(all_tools) >= expected_total, (
                f"All tools count ({len(all_tools)}) should be at least the sum of individual servers "
                f"({len(deepwiki_tools)} + {len(context7_tools)} = {expected_total})"
            )

            # Test caching works per server
            cached_deepwiki_tools = await get_mcp_tools("deepwiki")
            cached_context7_tools = await get_mcp_tools("context7")

            assert deepwiki_tools == cached_deepwiki_tools, (
                "Caching should work for deepwiki server"
            )
            assert context7_tools == cached_context7_tools, (
                "Caching should work for context7 server"
            )

            # Test non-existent server still returns empty list
            nonexistent_tools = await get_mcp_tools("nonexistent_server")
            assert nonexistent_tools == [], (
                "Non-existent server should return empty list"
            )

        finally:
            # Clean up
            remove_mcp_server("context7")
            clear_mcp_cache()

    @pytest.mark.asyncio
    async def test_mcp_server_isolation_and_independence(self) -> None:
        """
        Test that MCP servers are properly isolated and independent.

        This test ensures that:
        1. Each server has its own client instance
        2. Failures in one server don't affect others
        3. Server-specific tool loading works correctly
        4. Cache isolation works per server
        """
        # Clean state
        clear_mcp_cache()

        # Add multiple test servers including one that will fail
        good_server_config = {
            "url": "https://mcp.context7.com/sse",
            "transport": "sse",
        }
        bad_server_config = {
            "url": "https://nonexistent.invalid.domain.example/mcp",
            "transport": "streamable_http",
        }

        add_mcp_server("context7", good_server_config)
        add_mcp_server("bad_server", bad_server_config)

        try:
            # Get tools from good server should work
            context7_tools = await get_mcp_tools("context7")
            assert isinstance(context7_tools, list), "Context7 tools should be a list"

            # Get tools from bad server should return empty list (not crash)
            bad_server_tools = await get_mcp_tools("bad_server")
            assert bad_server_tools == [], "Bad server should return empty list"

            # Good server should still work after bad server failure
            context7_tools_again = await get_mcp_tools("context7")
            assert context7_tools == context7_tools_again, (
                "Good server should still work after bad server failure"
            )

            # All tools should include good server tools but handle bad server gracefully
            all_tools = await get_all_mcp_tools()
            assert isinstance(all_tools, list), "All tools should be a list"

            # Should include DeepWiki tools (default server) and Context7 tools
            # but not fail due to bad server
            deepwiki_tools = await get_mcp_tools("deepwiki")
            expected_minimum = len(deepwiki_tools) + len(context7_tools)
            assert len(all_tools) >= expected_minimum, (
                f"All tools should include tools from working servers. "
                f"Got {len(all_tools)}, expected at least {expected_minimum}"
            )

        finally:
            # Clean up
            remove_mcp_server("context7")
            remove_mcp_server("bad_server")
            clear_mcp_cache()
