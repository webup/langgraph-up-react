#!/usr/bin/env python3
"""
直接调用图的高级示例
展示更多高级功能和配置选项
"""
import asyncio
from typing import Dict, Any
from common.context import Context
from react_agent import graph


async def custom_model_example():
    """自定义模型配置示例"""
    print("=== 自定义模型配置示例 ===")
    
    # 使用不同的模型（如果有相应API密钥）
    models_to_try = [
        "qwen:qwen-flash",
        "qwen:qwen-plus", 
        # "openai:gpt-4o-mini",  # 需要OPENAI_API_KEY
        # "anthropic:claude-3.5-haiku",  # 需要ANTHROPIC_API_KEY
    ]
    
    for model in models_to_try:
        try:
            result = await graph.ainvoke(
                {"messages": [("user", "你好，请简单介绍一下自己")]},
                context=Context(
                    model=model,
                    system_prompt="你是一个友好的AI助手。"
                )
            )
            print(f"模型 {model}: {result['messages'][-1].content}")
        except Exception as e:
            print(f"模型 {model} 调用失败: {e}")
    print()


async def deepwiki_tools_example():
    """DeepWiki工具示例（如果启用）"""
    print("=== DeepWiki工具示例 ===")
    
    try:
        result = await graph.ainvoke(
            {"messages": [("user", "请帮我查询LangGraph项目的文档信息")]},
            context=Context(
                model="qwen:qwen-flash",
                system_prompt="你是一个AI助手，可以使用DeepWiki工具查询项目文档。",
                enable_deepwiki=True  # 启用DeepWiki工具
            )
        )
        print(f"DeepWiki查询结果: {result['messages'][-1].content}")
    except Exception as e:
        print(f"DeepWiki示例跳过: {e}")
    print()


async def step_by_step_execution():
    """逐步执行示例 - 查看每个节点的输出"""
    print("=== 逐步执行示例 ===")
    
    question = "Python中列表和元组的区别是什么？"
    print(f"问题: {question}")
    print("执行过程:")
    
    step = 1
    async for chunk in graph.astream(
        {"messages": [("user", question)]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个Python专家，请详细解答问题。"
        )
    ):
        for node_name, node_output in chunk.items():
            print(f"步骤 {step} - 节点 '{node_name}':")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        print(f"  内容: {msg.content[:100]}...")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"  工具调用: {len(msg.tool_calls)} 个")
            step += 1
    print()


async def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    try:
        # 测试没有API密钥的情况
        result = await graph.ainvoke(
            {"messages": [("user", "搜索最新的AI新闻")]},
            context=Context(
                model="invalid:model",  # 无效模型
                system_prompt="你是一个AI助手。"
            )
        )
    except Exception as e:
        print(f"预期的错误（无效模型）: {type(e).__name__}: {e}")
    
    try:
        # 正常调用作为对比
        result = await graph.ainvoke(
            {"messages": [("user", "1+1等于几？")]},
            context=Context(
                model="qwen:qwen-flash",
                system_prompt="你是一个数学助手。"
            )
        )
        print(f"正常调用成功: {result['messages'][-1].content}")
    except Exception as e:
        print(f"正常调用也失败: {e}")
    print()


async def batch_processing_example():
    """批量处理示例"""
    print("=== 批量处理示例 ===")
    
    questions = [
        "什么是机器学习？",
        "Python的主要特点是什么？",
        "解释一下递归的概念"
    ]
    
    tasks = []
    for i, question in enumerate(questions):
        task = graph.ainvoke(
            {"messages": [("user", question)]},
            context=Context(
                model="qwen:qwen-flash",
                system_prompt=f"你是AI助手#{i+1}，请简洁回答问题。"
            )
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, (question, result) in enumerate(zip(questions, results)):
        print(f"问题 {i+1}: {question}")
        if isinstance(result, Exception):
            print(f"  错误: {result}")
        else:
            print(f"  回答: {result['messages'][-1].content}")
        print()


async def main():
    """主函数"""
    print("LangGraph ReAct智能体高级调用示例\n")
    
    try:
        await custom_model_example()
        await deepwiki_tools_example()
        await step_by_step_execution()
        await error_handling_example()
        await batch_processing_example()
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("\n请检查：")
        print("1. 环境配置: cp .env.example .env")
        print("2. API密钥设置")
        print("3. 依赖安装: uv sync --dev")


if __name__ == "__main__":
    asyncio.run(main())