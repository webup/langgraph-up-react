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


async def test_api_simple_question() -> None:
    """Test agent can answer a simple question via API without tool usage."""
    client = get_client(url=LANGGRAPH_URL)
    assistant_id = await get_assistant_id()

    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Send a simple math question
    input_data = {"messages": [{"role": "human", "content": "What is 2 + 2?"}]}

    # Use client.runs.wait for non-streaming execution
    result = await client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input_data
    )

    # Verify we got a response
    assert result is not None
    messages = result["messages"]  # type: ignore
    assert len(messages) >= 2  # User message + AI response

    # Check that the response contains "4"
    ai_message = messages[-1]
    content = ai_message["content"]  # type: ignore
    assert "4" in str(content)


async def test_api_streaming_with_search() -> None:
    """Test agent can stream responses and use search tools via API."""
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

    # Stream the execution
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode="updates",
    ):
        chunks.append(chunk)

        # Check for tool usage in the chunk data
        if chunk.data and isinstance(chunk.data, dict):
            for node_name, node_data in chunk.data.items():
                if isinstance(node_data, dict):
                    # Check for tool_calls in messages
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get("tool_calls"):
                                    tool_calls_detected = True
                    # Direct tool_calls check
                    if node_data.get("tool_calls"):
                        tool_calls_detected = True
                # Check if we're in the tools node (indicates tool usage)
                if node_name == "tools" and node_data:
                    tool_calls_detected = True

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

    # Verify tool usage occurred
    assert tool_calls_detected, "Agent should have used search tool for this query"


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
