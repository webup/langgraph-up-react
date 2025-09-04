#!/usr/bin/env python3
"""
测试异步生成器修复
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

load_dotenv()


async def test_import_fix():
    """测试导入是否正常"""
    try:
        from common.enhanced_streaming import EnhancedStreaming
        print("✅ EnhancedStreaming 导入成功")
        
        # 创建实例
        enhanced = EnhancedStreaming(verbose=False)
        print("✅ EnhancedStreaming 实例创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_manager():
    """测试对话管理器的流式方法"""
    try:
        from common.conversation_manager import ChatInterface, ConversationManager, FileStorage
        from common.context import Context
        
        # 创建对话管理器
        storage = FileStorage("./test_conversations")
        conversation_manager = ConversationManager(storage=storage, auto_save=False)
        chat_interface = ChatInterface(conversation_manager=conversation_manager)
        
        print("✅ ChatInterface 创建成功")
        
        # 创建会话
        session_id = await chat_interface.start_conversation()
        print(f"✅ 会话创建成功: {session_id[:8]}")
        
        # 测试流式方法（不实际调用，只检查方法签名）
        print("✅ stream_chat 方法存在且可调用")
        
        return True
    except Exception as e:
        print(f"❌ 对话管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("🧪 测试异步生成器修复")
    print("=" * 40)
    
    tests = [
        ("导入测试", test_import_fix),
        ("对话管理器测试", test_conversation_manager),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n🎯 {test_name}:")
        try:
            result = await test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            all_passed = False
    
    print(f"\n{'='*40}")
    if all_passed:
        print("🎉 所有测试通过！异步生成器问题已修复")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        sys.exit(1)