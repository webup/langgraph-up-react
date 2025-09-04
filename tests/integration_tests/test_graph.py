from common.context import Context
from react_agent import graph


async def test_react_agent_simple_passthrough() -> None:
    """Test that the agent can answer a simple question about LangChain founder."""
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},  # type: ignore
        context=Context(
            model="siliconflow:Qwen/Qwen3-8B",
            system_prompt="You are a helpful AI assistant.",
        ),
    )

    # Should have user message + AI response
    assert len(res["messages"]) >= 2

    # Check that the response mentions Harrison (LangChain founder)
    final_response = str(res["messages"][-1].content).lower()
    assert "harrison" in final_response
