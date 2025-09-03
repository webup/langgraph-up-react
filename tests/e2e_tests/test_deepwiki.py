"""Refined end-to-end tests for DeepWiki MCP functionality with strict assertions."""

import pytest

from ..conftest import TEST_MODEL


@pytest.fixture
async def assistant_deepwiki_disabled(langgraph_client):
    """Create an assistant with deepwiki explicitly disabled."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={
            "model": TEST_MODEL,
            "enable_deepwiki": False,
            "system_prompt": "You are a helpful AI assistant.",
        },
    )
    return assistant["assistant_id"]


@pytest.fixture
async def assistant_deepwiki_enabled(langgraph_client):
    """Create an assistant with deepwiki explicitly enabled."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={
            "model": TEST_MODEL,
            "enable_deepwiki": True,
            "system_prompt": "You are a helpful AI assistant with access to deepwiki tools. When asked to use deepwiki tools, you must use them to get current documentation.",
        },
    )
    return assistant["assistant_id"]


class TestDeepWikiStrictE2E:
    """Strict end-to-end tests for DeepWiki functionality with no workarounds."""

    @pytest.mark.asyncio
    async def test_deepwiki_disabled_should_not_use_deepwiki_tools(
        self, langgraph_client, assistant_deepwiki_disabled
    ) -> None:
        """Test that deepwiki tools are NOT used when explicitly disabled."""
        # Create a new thread
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Ask a question that would normally trigger deepwiki but with deepwiki disabled
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "Look up React documentation. What are React hooks?",
                }
            ]
        }

        # Track execution
        chunks = []
        deepwiki_tool_used = False

        async for chunk in langgraph_client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_deepwiki_disabled,
            input=input_data,
            stream_mode="updates",
        ):
            chunks.append(chunk)

            if chunk.data and isinstance(chunk.data, dict):
                for node_name, node_data in chunk.data.items():
                    # Check for tool usage
                    if node_name == "tools" and node_data:
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if isinstance(messages, list):
                                for msg in messages:
                                    if (
                                        isinstance(msg, dict)
                                        and msg.get("type") == "tool"
                                    ):
                                        tool_name = str(msg.get("name", ""))
                                        if any(
                                            deepwiki_tool in tool_name
                                            for deepwiki_tool in [
                                                "read_wiki",
                                                "ask_question",
                                                "read_wiki_structure",
                                                "read_wiki_contents",
                                            ]
                                        ):
                                            deepwiki_tool_used = True

                    # Check for tool calls in messages
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get("tool_calls"):
                                    for tool_call in msg.get("tool_calls", []):
                                        if isinstance(tool_call, dict):
                                            tool_name = tool_call.get("name", "")
                                            if any(
                                                deepwiki_tool in tool_name
                                                for deepwiki_tool in [
                                                    "read_wiki",
                                                    "ask_question",
                                                    "read_wiki_structure",
                                                    "read_wiki_contents",
                                                ]
                                            ):
                                                deepwiki_tool_used = True

        # Get final state
        final_state = await langgraph_client.threads.get_state(thread_id)
        messages = final_state["values"]["messages"]

        # STRICT ASSERTIONS - No workarounds
        assert not deepwiki_tool_used, (
            "DeepWiki tools should NOT be used when enable_deepwiki=False"
        )

        # Should still provide a response (may use web_search or knowledge)
        assert len(messages) >= 2, "Should have user message and AI response"
        final_response = str(messages[-1]["content"])
        assert len(final_response) > 20, "Should provide a meaningful response"

    @pytest.mark.asyncio
    async def test_deepwiki_enabled_must_use_deepwiki_tools(
        self, langgraph_client, assistant_deepwiki_enabled
    ) -> None:
        """Test that deepwiki tools MUST be used when enabled and explicitly requested."""
        # Create a new thread
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Explicitly request deepwiki usage with a clear instruction
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "Use the deepwiki tools to look up React documentation and tell me what React hooks are. You MUST use deepwiki tools - do not give a generic answer.",
                }
            ]
        }

        # Track execution strictly
        chunks = []
        deepwiki_tool_planned = False
        deepwiki_tool_executed = False
        tool_calls_detected = False
        deepwiki_tool_names_used = []

        async for chunk in langgraph_client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_deepwiki_enabled,
            input=input_data,
            stream_mode="updates",
        ):
            chunks.append(chunk)

            if chunk.data and isinstance(chunk.data, dict):
                for node_name, node_data in chunk.data.items():
                    # Check for tool execution (tools node)
                    if node_name == "tools" and node_data:
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if isinstance(messages, list):
                                for msg in messages:
                                    if (
                                        isinstance(msg, dict)
                                        and msg.get("type") == "tool"
                                    ):
                                        tool_name = str(msg.get("name", ""))
                                        if any(
                                            deepwiki_tool in tool_name
                                            for deepwiki_tool in [
                                                "read_wiki",
                                                "ask_question",
                                                "read_wiki_structure",
                                                "read_wiki_contents",
                                            ]
                                        ):
                                            deepwiki_tool_executed = True
                                            deepwiki_tool_names_used.append(tool_name)

                    # Check for tool planning (call_model node)
                    if node_name == "call_model" and node_data:
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if isinstance(messages, list):
                                for msg in messages:
                                    if isinstance(msg, dict) and msg.get("tool_calls"):
                                        tool_calls_detected = True
                                        for tool_call in msg.get("tool_calls", []):
                                            if isinstance(tool_call, dict):
                                                tool_name = tool_call.get("name", "")
                                                if any(
                                                    deepwiki_tool in tool_name
                                                    for deepwiki_tool in [
                                                        "read_wiki",
                                                        "ask_question",
                                                        "read_wiki_structure",
                                                        "read_wiki_contents",
                                                    ]
                                                ):
                                                    deepwiki_tool_planned = True

        # Get final state
        final_state = await langgraph_client.threads.get_state(thread_id)
        messages = final_state["values"]["messages"]

        # STRICT ASSERTIONS - No fallbacks or workarounds
        assert len(chunks) > 0, "Should receive streaming chunks"

        assert tool_calls_detected, (
            "Tool calls should be detected when deepwiki is enabled and requested"
        )

        assert deepwiki_tool_planned, (
            "DeepWiki tools must be planned when explicitly requested. "
            "enable_deepwiki=True was set but no deepwiki tool calls were planned."
        )

        assert deepwiki_tool_executed, (
            f"DeepWiki tools must be executed when explicitly requested. "
            f"Tools used: {deepwiki_tool_names_used}. "
            f"This indicates either: 1) MCP client not working, 2) Tools not loaded, "
            f"3) Agent not following instructions to use deepwiki tools."
        )

        # Verify response quality
        final_response = str(messages[-1]["content"])
        assert len(final_response) > 50, (
            "Should provide substantial response when using deepwiki tools"
        )

        # Should mention React hooks (the requested topic)
        final_response_lower = final_response.lower()
        assert any(
            keyword in final_response_lower for keyword in ["react", "hook", "hooks"]
        ), f"Response should mention React hooks: {final_response[:200]}..."

        # Verify tool usage in conversation history
        tool_messages = [
            msg
            for msg in messages
            if isinstance(msg, dict) and msg.get("type") == "tool"
        ]
        assert len(tool_messages) > 0, (
            "Conversation should include tool result messages from deepwiki"
        )

        # Verify at least one tool message is from deepwiki
        deepwiki_tool_results = [
            msg
            for msg in tool_messages
            if isinstance(msg, dict)
            and any(
                deepwiki_tool in str(msg.get("name", ""))
                for deepwiki_tool in [
                    "read_wiki",
                    "ask_question",
                    "read_wiki_structure",
                    "read_wiki_contents",
                ]
            )
        ]
        assert len(deepwiki_tool_results) > 0, (
            f"Should have deepwiki tool result messages. "
            f"Tool messages found: {[msg.get('name') for msg in tool_messages if isinstance(msg, dict)]}"
        )

    @pytest.mark.asyncio
    async def test_deepwiki_server_availability_check(
        self, langgraph_client, assistant_id
    ) -> None:
        """Test to verify deepwiki server is accessible before running strict tests."""
        # This is a prerequisite test - if this fails, the environment is not ready
        from common.mcp import get_deepwiki_tools

        # Check if deepwiki tools can be loaded
        tools = await get_deepwiki_tools()

        assert len(tools) > 0, (
            "DeepWiki MCP server must be accessible for e2e tests to run. "
            "This is an environment issue, not a test failure. "
            f"Tools loaded: {len(tools)}"
        )

        # Verify expected tool names
        tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
        expected_tools = ["read_wiki_structure", "read_wiki_contents", "ask_question"]

        for expected_tool in expected_tools:
            assert any(expected_tool in name for name in tool_names), (
                f"Expected deepwiki tool '{expected_tool}' not found. "
                f"Available tools: {tool_names}"
            )

    @pytest.mark.asyncio
    async def test_deepwiki_configuration_persistence_e2e(
        self, langgraph_client, assistant_id
    ) -> None:
        """Test that deepwiki configuration persists throughout execution."""
        # Create a new thread
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Test with deepwiki enabled configuration
        input_data = {
            "messages": [{"role": "human", "content": "Hello, please keep it brief"}],
        }

        # This should execute without errors even if deepwiki tools aren't used
        result = await langgraph_client.runs.wait(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input_data,
        )

        # Basic validation
        messages = result["messages"]
        assert len(messages) >= 2, "Should have user message and response"

        final_response = str(messages[-1]["content"])
        assert len(final_response) > 0, "Should have non-empty response"
