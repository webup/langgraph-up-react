"""End-to-end tests for the LangGraph ReAct agent API.

These tests verify the complete agent workflow through the LangGraph API including:
- SDK client functionality
- Thread and run management
- Message streaming
- Tool usage via API
- Context handling through configuration
"""

import asyncio

import pytest
from langgraph_sdk import get_client

# Test configuration
LANGGRAPH_URL = "http://localhost:2024"


async def get_assistant_id() -> str:
    """Get the first available assistant ID."""
    client = get_client(url=LANGGRAPH_URL)
    assistants = await client.assistants.search()
    if not assistants:
        raise RuntimeError("No assistants found")
    return assistants[0]["assistant_id"]  # type: ignore


async def test_api_simple_question(
    langgraph_client, assistant_id, test_helpers
) -> None:
    """Test agent can answer a simple question via API without tool usage."""
    # Create a new thread
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    # Send a simple math question using centralized test data
    from tests.test_data import TestQuestions

    question_data = TestQuestions.SIMPLE_MATH
    input_data = {"messages": [{"role": "human", "content": question_data["question"]}]}

    # Use client.runs.wait for non-streaming execution
    result = await langgraph_client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input_data
    )

    # Verify we got a valid response using enhanced assertions
    assert result is not None, "Should receive a response from the agent"
    messages = result["messages"]  # type: ignore

    # Use enhanced helper for validation
    test_helpers.assert_valid_response(
        messages, question_data["expected_answer"], min_messages=2
    )

    # Additional validation for non-tool usage
    assert not any(
        msg.get("tool_calls") for msg in messages if isinstance(msg, dict)
    ), "Simple math question should not require tool usage"


async def test_api_streaming_with_search() -> None:
    """Test agent can stream responses and use search tools via API with detailed tracking."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Ask a question that should trigger search
    input_data = {
        "messages": [{"role": "human", "content": "Who is the founder of LangChain?"}]
    }

    chunks = []
    tool_calls_detected = False
    model_calls_detected = False
    search_tool_used = False

    # Stream the execution
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode="updates",
    ):
        chunks.append(chunk)

        # Detailed analysis of the chunk data
        if chunk.data and isinstance(chunk.data, dict):
            for node_name, node_data in chunk.data.items():
                # Track model calls (reasoning phase)
                if node_name == "call_model" and node_data:
                    model_calls_detected = True

                # Track tool execution (action phase)
                if node_name == "tools" and node_data:
                    tool_calls_detected = True
                    # Check if web_search tool was specifically used
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if (
                                    isinstance(msg, dict)
                                    and msg.get("type") == "tool"
                                    and "web_search" in str(msg.get("name", ""))
                                ):
                                    search_tool_used = True

                # Check for tool calls in messages (planning phase)
                if isinstance(node_data, dict) and "messages" in node_data:
                    messages = node_data["messages"]
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("tool_calls"):
                                tool_calls_detected = True
                                # Check if web_search tool was planned
                                for tool_call in msg.get("tool_calls", []):
                                    if (
                                        isinstance(tool_call, dict)
                                        and tool_call.get("name") == "web_search"
                                    ):
                                        search_tool_used = True

    # Verify we received streaming chunks
    assert len(chunks) > 0, "Should receive streaming chunks"

    # Get final state to check the result
    final_state = await client.threads.get_state(thread_id)
    messages = final_state["values"]["messages"]  # type: ignore

    # Check that the response mentions Harrison (LangChain founder)
    final_message = messages[-1]
    final_response = str(final_message["content"]).lower()  # type: ignore
    assert "harrison" in final_response, (
        f"Expected Harrison in response: {final_response}"
    )

    # Verify both model and tool interactions occurred
    assert model_calls_detected, "Agent should have made model calls (reasoning)"
    assert tool_calls_detected, "Agent should have used search tool (action)"
    assert search_tool_used, "Agent should have specifically used the web_search tool"

    # Verify the conversation flow includes tool messages
    tool_messages = [
        msg for msg in messages if isinstance(msg, dict) and msg.get("type") == "tool"
    ]
    assert len(tool_messages) > 0, "Conversation should include tool result messages"


async def test_api_thread_management() -> None:
    """Test thread creation, state management, and conversation flow via API."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # First message
    input1 = {"messages": [{"role": "human", "content": "My favorite color is blue."}]}

    await client.runs.wait(thread_id=thread_id, assistant_id=assistant_id, input=input1)

    # Second message in the same thread (should maintain context)
    input2 = {"messages": [{"role": "human", "content": "What is my favorite color?"}]}

    result2 = await client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input2
    )

    # Check that the agent remembered the color
    messages = result2["messages"]  # type: ignore
    final_response = str(messages[-1]["content"]).lower()  # type: ignore
    assert "blue" in final_response, (
        "Agent should remember the favorite color from conversation history"
    )


async def test_api_concurrent_runs() -> None:
    """Test handling multiple concurrent API requests."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create multiple threads
    thread1 = await client.threads.create()
    thread2 = await client.threads.create()

    # Define different inputs for each thread
    input1 = {"messages": [{"role": "human", "content": "What is 5 + 3?"}]}
    input2 = {"messages": [{"role": "human", "content": "What is 10 - 4?"}]}

    # Run both concurrently
    results = await asyncio.gather(
        client.runs.wait(
            thread_id=thread1["thread_id"], assistant_id=assistant_id, input=input1
        ),
        client.runs.wait(
            thread_id=thread2["thread_id"], assistant_id=assistant_id, input=input2
        ),
    )

    # Verify both got correct responses
    result1, result2 = results

    response1 = str(result1["messages"][-1]["content"])  # type: ignore
    response2 = str(result2["messages"][-1]["content"])  # type: ignore

    assert "8" in response1, f"Expected 8 in first response: {response1}"
    assert "6" in response2, f"Expected 6 in second response: {response2}"


async def test_api_error_handling() -> None:
    """Test API error handling with edge cases."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Test with minimal message (empty messages may be rejected)
    input_data = {"messages": [{"role": "human", "content": "Hi"}]}

    # Should handle minimal input gracefully
    result = await client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input_data
    )

    # Should still generate a response
    assert result is not None
    messages = result["messages"]  # type: ignore
    assert len(messages) >= 2
    assert len(str(messages[-1]["content"])) > 0  # type: ignore


async def test_api_assistants_list() -> None:
    """Test listing available assistants via API."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # List all assistants
    assistants = await client.assistants.search()

    # Should have at least our agent
    assert len(assistants) > 0, "Should have at least one assistant"

    # Find our agent assistant
    agent_found = False
    for assistant in assistants:
        if assistant["assistant_id"] == assistant_id:
            agent_found = True
            break

    assert agent_found, f"Should find assistant with ID '{assistant_id}'"


async def test_api_react_pattern_with_multiple_tools() -> None:
    """Test the full ReAct pattern: Reasoning -> Action (tools) -> Observation -> Reasoning."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Ask a complex question that should require multiple reasoning steps and tool usage
    input_data = {
        "messages": [
            {
                "role": "human",
                "content": "I need to research Python web frameworks. Can you search for information about FastAPI and Django, then compare them and recommend which one would be better for a beginner to learn?",
            }
        ]
    }

    chunks = []
    tool_calls_count = 0
    reasoning_steps = 0
    model_responses = 0

    # Stream the execution to observe the ReAct pattern
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode="updates",
    ):
        chunks.append(chunk)

        # Analyze the chunk to detect ReAct pattern components
        if chunk.data and isinstance(chunk.data, dict):
            for node_name, node_data in chunk.data.items():
                # Detect tool calls (Action step)
                if node_name == "tools" and node_data:
                    tool_calls_count += 1

                # Detect model reasoning (Reasoning step)
                if node_name == "call_model" and node_data:
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict):
                                    # Count messages with tool calls (reasoning before action)
                                    if msg.get("tool_calls"):
                                        reasoning_steps += 1
                                    # Count AI responses (reasoning after observation)
                                    if msg.get("type") == "ai" and msg.get("content"):
                                        model_responses += 1

    # Get final state to examine the complete conversation
    final_state = await client.threads.get_state(thread_id)
    messages = final_state["values"]["messages"]

    # Verify the ReAct pattern was followed
    assert len(chunks) > 0, "Should receive streaming chunks"
    assert tool_calls_count > 0, "Agent should have made tool calls (Action step)"
    assert len(messages) >= 3, "Should have user input, tool calls, and final response"

    # Check for comprehensive response that shows reasoning
    final_message = messages[-1]
    final_response = str(final_message["content"]).lower()

    # Verify the agent performed research and reasoning
    assert any(framework in final_response for framework in ["fastapi", "django"]), (
        "Response should mention the frameworks that were researched"
    )

    # Look for comparison/reasoning indicators
    reasoning_indicators = [
        "compare",
        "comparison",
        "better",
        "recommend",
        "beginner",
        "learn",
    ]
    assert any(indicator in final_response for indicator in reasoning_indicators), (
        f"Response should show reasoning/comparison: {final_response[:200]}..."
    )

    # Verify tool usage pattern in message history
    tool_usage_found = False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("tool_calls"):
            tool_usage_found = True
            break

    assert tool_usage_found, "Message history should contain tool calls"

    # Validate the ReAct pattern was successfully executed
    assert tool_calls_count > 0 and len(messages) > 2, (
        f"ReAct pattern should show tool usage: {tool_calls_count} tool calls, {len(messages)} messages"
    )


async def test_api_streaming_modes() -> None:
    """Test different streaming modes via API."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    input_data = {"messages": [{"role": "human", "content": "Hello, how are you?"}]}

    # Test different streaming modes
    for stream_mode in ["updates", "values"]:
        chunks = []

        async for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input_data,
            stream_mode=[stream_mode],  # type: ignore
        ):
            chunks.append(chunk)
            # Only process first few chunks to avoid overwhelming the test
            if len(chunks) >= 3:
                break

        assert len(chunks) > 0, f"Should receive chunks in {stream_mode} mode"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])
