#!/usr/bin/env python3
"""
多轮对话功能演示
展示新的对话管理功能的各种使用方式
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from src.common.context import Context
from src.common.conversation_manager import (
    ChatInterface,
    ConversationManager,
    quick_chat,
    quick_stream_chat,
    get_default_chat_interface,
)
from src.common.conversation import FileStorage, MemoryStorage

# 加载环境变量
load_dotenv()


async def basic_multi_turn_example():
    """基础多轮对话示例"""
    print("=== 基础多轮对话示例 ===")
    
    # 创建聊天接口
    chat = ChatInterface()
    
    # 开始对话
    session_id = await chat.start_conversation()
    print(f"开始对话，会话ID: {session_id}")
    
    # 多轮对话
    conversations = [
        "你好！我叫小明",
        "我最喜欢的颜色是蓝色",
        "我的名字是什么？",
        "我最喜欢的颜色是什么？",
        "请用我的名字和喜欢的颜色编一个小故事"
    ]
    
    for user_input in conversations:
        print(f"\n👤 用户: {user_input}")
        response = await chat.chat(user_input, session_id)
        print(f"🤖 AI: {response}")
    
    print(f"\n✅ 对话完成，会话ID: {session_id}")
    print()


async def persistent_conversation_example():
    """持久化对话示例"""
    print("=== 持久化对话示例 ===")
    
    # 使用文件存储确保持久化
    storage = FileStorage("./demo_conversations")
    conversation_manager = ConversationManager(storage=storage)
    chat = ChatInterface(conversation_manager)
    
    # 创建固定的会话ID
    session_id = "demo-persistent-session"
    
    print("第一阶段对话:")
    responses = []
    
    # 第一阶段对话
    for message in ["我正在学习Python编程", "我今天学了什么是函数"]:
        print(f"👤 用户: {message}")
        response = await chat.chat(message, session_id)
        responses.append(response)
        print(f"🤖 AI: {response}")
    
    print("\n--- 模拟程序重启 ---\n")
    
    # 创建新的聊天接口（模拟重启）
    storage2 = FileStorage("./demo_conversations")
    conversation_manager2 = ConversationManager(storage=storage2)
    chat2 = ChatInterface(conversation_manager2)
    
    print("第二阶段对话（从持久化恢复）:")
    
    # 继续对话，应该能记住之前的内容
    continue_message = "我刚才说我在学什么？"
    print(f"👤 用户: {continue_message}")
    response = await chat2.chat(continue_message, session_id)
    print(f"🤖 AI: {response}")
    
    # 显示对话历史
    history = await chat2.get_conversation_history(session_id)
    print(f"\n📋 对话历史 (共{len(history)}条消息):")
    for i, msg in enumerate(history[-6:], 1):  # 显示最后6条消息
        role = "👤" if msg["role"] == "human" else "🤖"
        print(f"  {i}. {role} {msg['content'][:50]}...")
    
    print()


async def streaming_multi_turn_example():
    """流式多轮对话示例"""
    print("=== 流式多轮对话示例 ===")
    
    chat = ChatInterface()
    session_id = await chat.start_conversation()
    
    conversations = [
        "请详细介绍什么是机器学习",
        "那深度学习和机器学习有什么区别？",
        "给我一个简单的代码示例"
    ]
    
    for user_input in conversations:
        print(f"\n👤 用户: {user_input}")
        print("🤖 AI: ", end="", flush=True)
        
        full_response = ""
        async for chunk in chat.stream_chat(user_input, session_id):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()  # 换行
    
    print()


async def quick_functions_example():
    """便捷函数使用示例"""
    print("=== 便捷函数使用示例 ===")
    
    # 使用 quick_chat 快速对话
    response1, session_id = await quick_chat("你好，我是新用户")
    print(f"👤 用户: 你好，我是新用户")
    print(f"🤖 AI: {response1}")
    print(f"📝 会话ID: {session_id}")
    
    # 继续使用相同会话
    response2, _ = await quick_chat("我刚才说了什么？", session_id)
    print(f"\n👤 用户: 我刚才说了什么？")
    print(f"🤖 AI: {response2}")
    
    # 使用流式快速对话
    print(f"\n👤 用户: 请介绍一下Python")
    print("🤖 AI (流式): ", end="", flush=True)
    
    stream, _ = await quick_stream_chat("请介绍一下Python", session_id)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    
    print("\n")


async def conversation_management_example():
    """对话管理功能示例"""
    print("=== 对话管理功能示例 ===")
    
    chat = ChatInterface()
    
    # 创建多个对话会话
    session_ids = []
    for i in range(3):
        session_id = await chat.start_conversation()
        session_ids.append(session_id)
        
        # 在每个会话中进行对话
        message = f"我是用户{i+1}，我喜欢{['音乐', '电影', '运动'][i]}"
        await chat.chat(message, session_id)
        print(f"创建会话 {i+1}: {session_id[:8]}...")
    
    # 列出所有对话
    all_sessions = await chat.list_conversations()
    print(f"\n📋 当前共有 {len(all_sessions)} 个对话会话")
    
    # 在不同会话中进行对话，验证上下文隔离
    for i, session_id in enumerate(session_ids[:2]):
        response = await chat.chat("我刚才说我喜欢什么？", session_id)
        print(f"会话 {i+1}: {response}")
    
    # 清理一个会话
    if session_ids:
        deleted = await chat.clear_conversation(session_ids[0])
        print(f"\n🗑️ 清理会话: {'成功' if deleted else '失败'}")
    
    # 再次列出对话
    remaining_sessions = await chat.list_conversations()
    print(f"清理后剩余 {len(remaining_sessions)} 个会话")
    
    print()


async def context_customization_example():
    """上下文定制示例"""
    print("=== 上下文定制示例 ===")
    
    # 使用不同的上下文配置
    contexts = [
        Context(
            model="qwen:qwen-flash",
            system_prompt="你是一个友好的Python编程导师，请用简单易懂的方式回答问题。"
        ),
        Context(
            model="qwen:qwen-flash", 
            system_prompt="你是一个严谨的学术专家，请用专业术语详细回答问题。"
        )
    ]
    
    chat = ChatInterface()
    
    question = "什么是递归？"
    
    for i, context in enumerate(contexts):
        session_id = await chat.start_conversation()
        print(f"\n情境 {i+1} ({'友好导师' if i == 0 else '学术专家'}):")
        print(f"👤 用户: {question}")
        
        response = await chat.chat(question, session_id, context)
        print(f"🤖 AI: {response[:200]}...")  # 只显示前200字符
    
    print()


async def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    chat = ChatInterface()
    
    # 测试无效会话ID
    try:
        response = await chat.chat("你好", "invalid-session-id")
        print(f"使用无效会话ID: {response[:50]}...")
    except Exception as e:
        print(f"预期的错误: {e}")
    
    # 测试正常流程的健壮性
    session_id = await chat.start_conversation()
    
    try:
        # 正常对话
        response = await chat.chat("测试消息", session_id)
        print(f"正常对话: {response[:50]}...")
        
        # 获取对话历史
        history = await chat.get_conversation_history(session_id)
        print(f"对话历史: 共{len(history)}条消息")
        
    except Exception as e:
        print(f"意外错误: {e}")
    
    print()


async def main():
    """主函数 - 运行所有示例"""
    print("🚀 LangGraph ReAct 多轮对话功能演示\n")
    
    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 错误：未找到 DASHSCOPE_API_KEY")
        print("请确保 .env 文件存在并包含正确的API密钥")
        return
    else:
        print(f"✅ API密钥已配置: {api_key[:10]}...")
        print()
    
    try:
        # 运行各种示例
        await basic_multi_turn_example()
        await persistent_conversation_example() 
        
        # 可选的其他示例（注释掉以避免过多输出）
        # await streaming_multi_turn_example()
        # await quick_functions_example()
        # await conversation_management_example()
        # await context_customization_example()
        # await error_handling_example()
        
        print("🎉 所有示例运行完成！")
        print("\n📚 功能总结:")
        print("✅ 多轮对话记忆")
        print("✅ 会话持久化存储")
        print("✅ 对话历史管理")
        print("✅ 流式对话支持")
        print("✅ 便捷API接口")
        print("✅ 自定义上下文")
        print("✅ 错误处理机制")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("\n解决方案：")
        print("1. 配置环境: cp .env.example .env")
        print("2. 设置API密钥（至少需要DASHSCOPE_API_KEY）")
        print("3. 安装依赖: uv sync --dev")


if __name__ == "__main__":
    asyncio.run(main())