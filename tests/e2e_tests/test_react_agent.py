"""End-to-end tests for the LangGraph ReAct agent API."""


async def test_api_simple_question(
    langgraph_client, assistant_id, test_helpers
) -> None:
    """Test agent can answer a simple question without tool usage."""
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    from tests.test_data import TestQuestions

    question_data = TestQuestions.SIMPLE_MATH
    input_data = {"messages": [{"role": "human", "content": question_data["question"]}]}

    result = await langgraph_client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input_data
    )

    messages = result["messages"]
    test_helpers.assert_valid_response(
        messages, question_data["expected_answer"], min_messages=2
    )
    assert not any(
        msg.get("tool_calls") for msg in messages if isinstance(msg, dict)
    ), "Simple math question should not require tool usage"


async def test_api_streaming_with_search(langgraph_client, assistant_id) -> None:
    """Test agent streaming and tool usage in ReAct pattern."""
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    input_data = {
        "messages": [
            {
                "role": "human",
                "content": "What was the latest LangChain release version announced in December 2024?",
            }
        ]
    }

    chunks = []
    tool_calls_detected = False

    async for chunk in langgraph_client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode="updates",
    ):
        chunks.append(chunk)
        if chunk.data and isinstance(chunk.data, dict):
            for node_name, node_data in chunk.data.items():
                if node_name == "tools" and node_data:
                    tool_calls_detected = True

    assert len(chunks) > 0, "Should receive streaming chunks"
    assert tool_calls_detected, "Agent should have used search tool"

    final_state = await langgraph_client.threads.get_state(thread_id)
    messages = final_state["values"]["messages"]
    final_response = str(messages[-1]["content"]).lower()
    assert any(
        keyword in final_response for keyword in ["version", "release", "langchain"]
    ), f"Expected version/release info in response: {final_response}"


async def test_api_thread_management(langgraph_client, assistant_id) -> None:
    """Test conversation context persistence across messages."""
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    # First message
    input1 = {"messages": [{"role": "human", "content": "My favorite color is blue."}]}
    await langgraph_client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input1
    )

    # Second message (should maintain context)
    input2 = {"messages": [{"role": "human", "content": "What is my favorite color?"}]}
    result2 = await langgraph_client.runs.wait(
        thread_id=thread_id, assistant_id=assistant_id, input=input2
    )

    messages = result2["messages"]
    final_response = str(messages[-1]["content"]).lower()
    assert "blue" in final_response, (
        "Agent should remember context from conversation history"
    )


async def test_api_react_pattern_with_multiple_tools(
    langgraph_client, assistant_id
) -> None:
    """Test the full ReAct pattern: Reasoning -> Action -> Observation."""
    thread = await langgraph_client.threads.create()
    thread_id = thread["thread_id"]

    input_data = {
        "messages": [
            {
                "role": "human",
                "content": "I need to research Python web frameworks. Can you search for information about FastAPI and Django, then compare them?",
            }
        ]
    }

    chunks = []
    tool_calls_count = 0

    async for chunk in langgraph_client.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input=input_data,
        stream_mode="updates",
    ):
        chunks.append(chunk)
        if chunk.data and isinstance(chunk.data, dict):
            for node_name, node_data in chunk.data.items():
                if node_name == "tools" and node_data:
                    tool_calls_count += 1

    final_state = await langgraph_client.threads.get_state(thread_id)
    messages = final_state["values"]["messages"]
    final_response = str(messages[-1]["content"]).lower()

    assert len(chunks) > 0, "Should receive streaming chunks"
    assert tool_calls_count > 0, "Agent should have made tool calls"
    assert any(framework in final_response for framework in ["fastapi", "django"]), (
        "Response should mention the frameworks that were researched"
    )
