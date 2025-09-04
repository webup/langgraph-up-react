"""Comprehensive E2E tests for SiliconFlow integration."""

import os

import pytest

# Test Models
QWEN3_8B = "siliconflow:Qwen/Qwen3-8B"  # Traditional text model
GLM_Z1 = (
    "siliconflow:THUDM/GLM-Z1-9B-0414"  # Reasoning model (supports function calling)
)

# Test Data
STUDIO_UI_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "static", "studio_ui.png"
)


# ===== TRADITIONAL MODEL TESTS (Qwen/Qwen3-8B) =====


@pytest.mark.e2e
async def test_qwen3_basic_text_generation(langgraph_client, test_helpers) -> None:
    """Test basic text generation with Qwen3-8B."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={"model": QWEN3_8B},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Test simple arithmetic
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "What is 2+2? Answer only with the number.",
                }
            ]
        }

        result = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input_data
        )

        messages = result["messages"]
        test_helpers.assert_valid_response(messages, "4", min_messages=2)

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_qwen3_tool_calling_web_search(langgraph_client) -> None:
    """Test Qwen3-8B with web search tool calling."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={"model": QWEN3_8B},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Ask about recent information requiring web search
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "What was the latest major Python release in 2024? Please search for current information.",
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
        assert tool_calls_detected, "Agent should have made tool calls"

        # Verify response mentions Python and version
        final_state = await langgraph_client.threads.get_state(thread_id)
        messages = final_state["values"]["messages"]
        final_response = str(messages[-1]["content"]).lower()
        assert any(term in final_response for term in ["python", "3.1", "release"]), (
            f"Expected Python release info: {final_response}"
        )

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_qwen3_multi_turn_conversation(langgraph_client) -> None:
    """Test Qwen3-8B multi-turn conversation with context retention."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        name="Qwen3 Context Test",
        description="Test context retention",
        context={"model": QWEN3_8B},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # First message - establish context
        input1 = {
            "messages": [
                {
                    "role": "human",
                    "content": "I'm working on a Python project about data analysis.",
                }
            ]
        }
        await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input1
        )

        # Second message - reference previous context
        input2 = {
            "messages": [
                {
                    "role": "human",
                    "content": "What libraries would be most useful for my project?",
                }
            ]
        }
        result2 = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input2
        )

        messages = result2["messages"]
        final_response = str(messages[-1]["content"]).lower()

        # Should mention relevant data analysis libraries
        assert any(
            lib in final_response for lib in ["pandas", "numpy", "matplotlib"]
        ), f"Expected data analysis libraries: {final_response}"

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_qwen3_streaming_responses(langgraph_client) -> None:
    """Test Qwen3-8B streaming functionality."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        name="Qwen3 Streaming Test",
        description="Test streaming responses",
        context={"model": QWEN3_8B},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "Write a short poem about artificial intelligence.",
                }
            ]
        }

        chunks = []
        async for chunk in langgraph_client.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input_data,
            stream_mode="updates",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive streaming chunks"

        # Verify final response is a poem
        final_state = await langgraph_client.threads.get_state(thread_id)
        messages = final_state["values"]["messages"]
        final_response = str(messages[-1]["content"]).lower()
        assert len(final_response) > 50, "Should have a substantial poem"
        assert any(
            term in final_response for term in ["ai", "intelligence", "machine"]
        ), f"Expected AI-related terms: {final_response}"

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_qwen3_regional_endpoints(langgraph_client, test_helpers) -> None:
    """Test Qwen3-8B with different regional configurations."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        name="Qwen3 Regional Test",
        description="Test regional endpoint handling",
        context={"model": QWEN3_8B},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "Hello! Please respond with 'Regional test successful'.",
                }
            ]
        }

        result = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input_data
        )

        messages = result["messages"]
        assert len(messages) >= 2, "Should have at least human and AI messages"
        final_response = str(messages[-1]["content"]).lower()
        assert "regional" in final_response or "successful" in final_response, (
            f"Expected confirmation message: {final_response}"
        )

    finally:
        await langgraph_client.assistants.delete(assistant_id)


# ===== REASONING MODEL TESTS (THUDM/GLM-Z1-9B-0414) =====


@pytest.mark.e2e
async def test_glm_z1_reasoning_task(langgraph_client) -> None:
    """Test GLM-Z1 with complex reasoning without tools."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={"model": GLM_Z1},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Test complex reasoning task
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "Solve this logic puzzle: A farmer has chickens and rabbits. In total, there are 35 heads and 94 feet. How many chickens and how many rabbits are there? Show your reasoning step by step.",
                }
            ]
        }

        result = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input_data
        )

        messages = result["messages"]
        assert len(messages) >= 2, "Should have human and AI messages"

        final_response = str(messages[-1]["content"]).lower()

        # Should show mathematical reasoning
        expected_terms = [
            "chicken",
            "rabbit",
            "equation",
            "solve",
            "feet",
            "heads",
        ]
        assert any(term in final_response for term in expected_terms), (
            f"Expected reasoning terms in response: {final_response[:200]}..."
        )

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_glm4_complex_reasoning_task(langgraph_client) -> None:
    """Test GLM-4.1V complex reasoning capabilities."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        context={"model": GLM_Z1},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Complex reasoning problem
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "A farmer has chickens and cows. There are 30 heads and 74 legs total. How many chickens and how many cows are there? Show your reasoning step by step.",
                }
            ]
        }

        result = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input_data
        )

        messages = result["messages"]
        final_response = str(messages[-1]["content"]).lower()

        # Should show step-by-step reasoning and correct answer
        assert "step" in final_response or "reasoning" in final_response, (
            f"Expected step-by-step reasoning: {final_response[:200]}..."
        )

        # The answer should be 22 chickens and 8 cows
        assert any(
            num in final_response for num in ["22", "8", "twenty-two", "eight"]
        ), f"Expected correct numbers in answer: {final_response}"

    finally:
        await langgraph_client.assistants.delete(assistant_id)


@pytest.mark.e2e
async def test_glm4_reasoning_chain(langgraph_client) -> None:
    """Test GLM-4.1V reasoning chain capabilities."""
    assistant = await langgraph_client.assistants.create(
        graph_id="agent",
        name="GLM4 Reasoning Chain Test",
        description="Test reasoning chains",
        context={"model": GLM_Z1},
    )
    assistant_id = assistant["assistant_id"]

    try:
        thread = await langgraph_client.threads.create()
        thread_id = thread["thread_id"]

        # Multi-step reasoning problem
        input_data = {
            "messages": [
                {
                    "role": "human",
                    "content": "If I have a 5x5 grid and I want to place queens so none attack each other (like in chess), is this possible? If so, how many solutions exist? Think through this systematically.",
                }
            ]
        }

        result = await langgraph_client.runs.wait(
            thread_id=thread_id, assistant_id=assistant_id, input=input_data
        )

        messages = result["messages"]
        final_response = str(messages[-1]["content"]).lower()

        # Should show systematic thinking and knowledge of N-Queens problem
        reasoning_indicators = ["systematic", "step", "consider", "analyze", "think"]
        has_reasoning = any(
            indicator in final_response for indicator in reasoning_indicators
        )

        queens_knowledge = any(
            term in final_response
            for term in ["queens", "attack", "diagonal", "solution"]
        )

        assert has_reasoning, (
            f"Expected systematic reasoning: {final_response[:200]}..."
        )
        assert queens_knowledge, (
            f"Expected N-Queens knowledge: {final_response[:200]}..."
        )

    finally:
        await langgraph_client.assistants.delete(assistant_id)
