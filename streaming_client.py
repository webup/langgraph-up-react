#!/usr/bin/env python3
"""
流式调用客户端示例
展示各种流式处理模式
"""
import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Any
from dotenv import load_dotenv
from common.context import Context
from react_agent import graph

# 显式加载.env文件
load_dotenv()


async def basic_streaming():
    """基础流式调用"""
    print("=== 基础流式调用 ===")
    
    question = "请详细解释什么是人工智能？"
    print(f"问题: {question}")
    print("回答: ", end="", flush=True)
    
    full_response = ""
    async for chunk in graph.astream(
        {"messages": [("user", question)]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个AI专家，请详细回答问题。"
        )
    ):
        # 处理每个节点的输出
        for node_name, node_output in chunk.items():
            if node_name == "call_model" and "messages" in node_output:
                message = node_output["messages"][-1]
                if hasattr(message, 'content') and message.content:
                    # 实时打印内容
                    print(message.content, end="", flush=True)
                    full_response = message.content
    
    print(f"\n\n完整回答: {full_response}\n")


async def streaming_with_tool_calls():
    """带工具调用的流式处理"""
    print("=== 带工具调用的流式处理 ===")
    
    question = "请搜索并告诉我最新的Python版本信息"
    print(f"问题: {question}")
    print("处理过程:")
    
    step = 1
    async for chunk in graph.astream(
        {"messages": [("user", question)]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个技术专家，可以使用搜索工具获取最新信息。"
        )
    ):
        for node_name, node_output in chunk.items():
            print(f"\n步骤 {step}: 节点 '{node_name}'")
            
            if "messages" in node_output:
                for message in node_output["messages"]:
                    # 处理AI消息
                    if hasattr(message, 'content') and message.content:
                        print(f"  💭 思考: {message.content[:100]}...")
                    
                    # 处理工具调用
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        for tool_call in message.tool_calls:
                            print(f"  🔧 调用工具: {tool_call.get('name', 'unknown')}")
                            print(f"     参数: {tool_call.get('args', {})}")
                    
                    # 处理工具结果
                    if hasattr(message, 'name'):  # ToolMessage
                        print(f"  📊 工具 '{message.name}' 结果: {str(message.content)[:100]}...")
            
            step += 1
    
    print()


async def streaming_with_interrupts():
    """带中断的流式处理"""
    print("=== 带中断的流式处理示例 ===")
    
    question = "请分步骤解释如何学习机器学习？"
    print(f"问题: {question}")
    print("回答 (可中断): ")
    
    count = 0
    async for chunk in graph.astream(
        {"messages": [("user", question)]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个教育专家，请分步骤详细回答。"
        )
    ):
        count += 1
        
        for node_name, node_output in chunk.items():
            if node_name == "call_model" and "messages" in node_output:
                message = node_output["messages"][-1]
                if hasattr(message, 'content') and message.content:
                    print(f"[块 {count}] {message.content}")
        
        # 模拟用户中断（在第3个块后停止）
        if count >= 3:
            print("\n[用户中断] 已获得足够信息，停止接收...\n")
            break


async def streaming_json_mode():
    """JSON格式流式输出"""
    print("=== JSON格式流式输出 ===")
    
    question = "请用JSON格式列出Python的5个主要特点"
    print(f"问题: {question}")
    print("JSON结果:")
    
    async for chunk in graph.astream(
        {"messages": [("user", question)]},
        context=Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个技术专家。请严格按照JSON格式回答，不要有其他文字。"
        )
    ):
        for node_name, node_output in chunk.items():
            if node_name == "call_model" and "messages" in node_output:
                message = node_output["messages"][-1]
                if hasattr(message, 'content') and message.content:
                    try:
                        # 尝试解析JSON
                        content = message.content.strip()
                        if content.startswith('{') or content.startswith('['):
                            parsed = json.loads(content)
                            print(json.dumps(parsed, indent=2, ensure_ascii=False))
                        else:
                            print(f"非JSON内容: {content}")
                    except json.JSONDecodeError:
                        print(f"JSON解析失败: {message.content}")
    
    print()


async def concurrent_streaming():
    """并发流式处理"""
    print("=== 并发流式处理 ===")
    
    questions = [
        "什么是深度学习？",
        "什么是自然语言处理？", 
        "什么是计算机视觉？"
    ]
    
    async def process_question(q: str, index: int):
        print(f"\n[线程 {index+1}] 问题: {q}")
        print(f"[线程 {index+1}] 回答: ", end="", flush=True)
        
        async for chunk in graph.astream(
            {"messages": [("user", q)]},
            context=Context(
                model="qwen:qwen-flash",
                system_prompt=f"你是AI专家#{index+1}，请简洁回答。"
            )
        ):
            for node_name, node_output in chunk.items():
                if node_name == "call_model" and "messages" in node_output:
                    message = node_output["messages"][-1]
                    if hasattr(message, 'content') and message.content:
                        print(f"[线程 {index+1}] {message.content}")
                        break
    
    # 并发执行
    tasks = [process_question(q, i) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    print()


async def custom_stream_handler():
    """自定义流处理器"""
    print("=== 自定义流处理器 ===")
    
    class CustomStreamHandler:
        def __init__(self):
            self.total_tokens = 0
            self.start_time = None
            self.responses = []
        
        async def handle_stream(self, question: str):
            import time
            self.start_time = time.time()
            
            print(f"🤖 开始处理: {question}")
            
            async for chunk in graph.astream(
                {"messages": [("user", question)]},
                context=Context(
                    model="qwen:qwen-flash",
                    system_prompt="你是一个helpful AI助手。"
                )
            ):
                await self.process_chunk(chunk)
            
            self.print_summary()
        
        async def process_chunk(self, chunk):
            for node_name, node_output in chunk.items():
                if "messages" in node_output:
                    for message in node_output["messages"]:
                        if hasattr(message, 'content') and message.content:
                            self.responses.append(message.content)
                            # 估算token数量（简单估算：字符数/4）
                            self.total_tokens += len(message.content) // 4
        
        def print_summary(self):
            import time
            duration = time.time() - self.start_time
            print(f"\n📊 处理摘要:")
            print(f"   - 响应数量: {len(self.responses)}")
            print(f"   - 估算Tokens: {self.total_tokens}")
            print(f"   - 处理时间: {duration:.2f}秒")
            if self.responses:
                print(f"   - 最终回答: {self.responses[-1][:100]}...")
    
    handler = CustomStreamHandler()
    await handler.handle_stream("解释一下量子计算的基本原理")
    
    print()


async def main():
    """主函数"""
    print("LangGraph ReAct智能体流式调用示例\n")
    
    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 错误：未找到 DASHSCOPE_API_KEY")
        print("请确保 .env 文件存在并包含正确的API密钥")
        return
    else:
        print(f"✅ API密钥已配置: {api_key[:10]}...")
    
    try:
        await basic_streaming()
        
        # 搜索工具示例（需要API密钥）
        try:
            await streaming_with_tool_calls()
        except Exception as e:
            print(f"工具调用示例跳过: {e}\n")
        
        await streaming_with_interrupts()
        await streaming_json_mode()
        await concurrent_streaming()
        await custom_stream_handler()
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("\n解决方案：")
        print("1. 配置环境: cp .env.example .env")
        print("2. 设置API密钥（至少需要DASHSCOPE_API_KEY用于Qwen模型）")
        print("3. 安装依赖: uv sync --dev")


if __name__ == "__main__":
    asyncio.run(main())