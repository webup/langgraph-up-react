#!/usr/bin/env python3
"""
API客户端调用示例
演示如何通过不同方式调用LangGraph ReAct智能体
"""
import asyncio
import os

from dotenv import load_dotenv

from common.context import Context
from react_agent import graph

# 显式加载.env文件
load_dotenv()


async def simple_question_example():
    """基础问答示例 - 不需要工具调用"""
    print("=== 基础问答示例 ===")
    
    result = await graph.ainvoke(
        {"messages": [("user", "今天北京天气怎么样？")]},
        context=Context(
            model="qwen:qwen-flash",  # 可以改为其他模型
            system_prompt="你是一个有用的AI助手。请简洁回答用户问题。"
        )
    )
    
    print("用户问题: 今天北京天气怎么样？")
    print(f"AI回答: {result['messages'][-1].content}")
    print()


async def search_question_example():
    """需要搜索工具的问题示例"""
    print("=== 搜索工具示例 ===")
    
    result = await graph.ainvoke(
        {"messages": [("user", "最新的Python 3.12版本有什么新特性？")]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个AI助手，可以使用搜索工具来获取最新信息。"
        )
    )
    
    print("用户问题: 最新的Python 3.12版本有什么新特性？")
    print(f"AI回答: {result['messages'][-1].content}")
    print()


async def streaming_example():
    """流式调用示例"""
    print("=== 流式调用示例 ===")
    
    print("用户问题: 请介绍一下LangGraph框架")
    print("AI回答(流式): ", end="", flush=True)
    
    async for chunk in graph.astream(
        {"messages": [("user", "请介绍一下LangGraph框架")]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个AI助手，请详细回答用户问题。"
        )
    ):
        # 打印每个节点的输出
        for node_name, node_output in chunk.items():
            if node_name == "call_model" and "messages" in node_output:
                content = node_output["messages"][-1].content
                if content:
                    print(content, end="", flush=True)
    
    print("\n")


async def conversation_example():
    """多轮对话示例"""
    print("=== 多轮对话示例 ===")
    
    # 初始状态
    state = {"messages": []}
    
    # 第一轮对话
    state = await graph.ainvoke(
        {"messages": [("user", "我最喜欢的颜色是蓝色")]},
        context=Context(model="qwen:qwen-flash")
    )
    print("用户: 我最喜欢的颜色是蓝色")
    print(f"AI: {state['messages'][-1].content}")
    
    # 第二轮对话（利用上下文）
    state["messages"].append(("user", "我最喜欢的颜色是什么？"))
    result = await graph.ainvoke(
        state,
        context=Context(model="qwen:qwen-flash")
    )
    print("用户: 我最喜欢的颜色是什么？")
    print(f"AI: {result['messages'][-1].content}")
    print()


async def main():
    """主函数 - 运行所有示例"""
    print("LangGraph ReAct智能体API调用示例\n")
    
    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 错误：未找到 DASHSCOPE_API_KEY")
        print("请确保 .env 文件存在并包含正确的API密钥")
        return
    else:
        print(f"✅ API密钥已配置: {api_key[:10]}...")
    
    try:
        # 1. 基础问答
        await simple_question_example()
        
        # # 2. 搜索工具使用（需要TAVILY_API_KEY）
        # try:
        #     await search_question_example()
        # except Exception as e:
        #     print(f"搜索示例跳过（可能缺少API密钥）: {e}\n")
        
        # # 3. 流式调用
        # await streaming_example()
        
        # # 4. 多轮对话
        # await conversation_example()
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保：")
        print("1. 已安装所有依赖: uv sync --dev")
        print("2. 已配置.env文件（从.env.example复制）")
        print("3. 已设置相应的API密钥")


if __name__ == "__main__":
    asyncio.run(main())