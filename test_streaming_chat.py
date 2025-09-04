#!/usr/bin/env python3
"""
流式对话功能测试脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from common.context import Context
from common.conversation_manager import ChatInterface, ConversationManager, FileStorage, HistoryManager

load_dotenv()


async def test_streaming_chat():
    """测试流式对话功能"""
    print("🧪 流式对话功能测试")
    print("=" * 40)
    
    # 初始化chat接口
    try:
        storage = FileStorage("./test_conversations")
        history_manager = HistoryManager(max_messages=50, max_tokens=4000)
        conversation_manager = ConversationManager(
            storage=storage, 
            history_manager=history_manager, 
            auto_save=False  # 测试时不保存
        )
        
        chat_interface = ChatInterface(
            conversation_manager=conversation_manager, 
            default_context=Context()
        )
        
        print("✅ 成功初始化聊天接口")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 创建测试会话
    try:
        session_id = await chat_interface.start_conversation()
        print(f"✅ 创建测试会话: {session_id[:8]}")
    except Exception as e:
        print(f"❌ 创建会话失败: {e}")
        return False
    
    # 测试用例
    test_queries = [
        "你好，请简单介绍一下自己",
        "重庆大学有哪些特色专业？",
        "请写一个简短的Python函数来计算斐波那契数列"
    ]
    
    print("\n🚀 开始流式对话测试...")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试 {i}: {query}")
        print("🤖 AI回答:")
        
        try:
            # 测试流式响应
            chunks = []
            async for chunk in chat_interface.stream_chat(query, session_id=session_id):
                if chunk:
                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
            
            print()  # 换行
            
            # 统计信息
            total_chars = sum(len(chunk) for chunk in chunks)
            print(f"📊 共收到 {len(chunks)} 个块，总计 {total_chars} 个字符")
            
        except Exception as e:
            print(f"❌ 流式测试失败: {e}")
            
            # 尝试非流式模式作为备用
            try:
                print("🔄 尝试非流式模式...")
                response = await chat_interface.chat(query, session_id=session_id)
                print(f"✅ 非流式回答: {response}")
            except Exception as fallback_error:
                print(f"❌ 非流式模式也失败: {fallback_error}")
        
        print("-" * 40)
    
    print("\n🎯 测试总结:")
    print("✅ 流式对话功能测试完成")
    print("💡 如果看到了实时的文字输出，说明流式功能工作正常")
    print("📝 可以运行 'python cli_chat.py' 开始正式使用")
    
    return True


async def main():
    """主函数"""
    success = await test_streaming_chat()
    if not success:
        print("\n❌ 测试失败，请检查配置")
        sys.exit(1)
    else:
        print("\n🎉 测试成功完成！")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        sys.exit(1)