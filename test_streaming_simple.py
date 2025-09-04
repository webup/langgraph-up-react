#!/usr/bin/env python3
"""
简单的流式功能测试
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from common.streaming_wrapper import StreamingWrapper, stream_text


async def test_streaming_wrapper():
    """测试流式包装器本身"""
    print("🧪 测试流式包装器")
    print("=" * 40)
    
    test_text = "你好！我是重庆大学AI助手。我可以帮助你了解重庆大学的各种信息，包括学校历史、专业设置、校园生活等。有什么问题尽管问我！"
    
    print("📝 原文本:")
    print(f'"{test_text}"')
    print()
    print("🌊 流式输出:")
    print('"', end='', flush=True)
    
    async for chunk in stream_text(test_text, chunk_size=3, base_delay=0.05):
        print(chunk, end='', flush=True)
    
    print('"')
    print()
    print("✅ 流式包装器测试完成")


async def test_mock_function():
    """测试包装普通函数"""
    print("\n🧪 测试函数包装")
    print("=" * 40)
    
    # 模拟一个返回文本的函数
    def mock_ai_response():
        return "重庆大学位于重庆市沙坪坝区，是教育部直属的全国重点大学。学校创建于1929年，是中国最早的现代大学之一。"
    
    print("🤖 模拟AI回答:")
    
    wrapper = StreamingWrapper(base_delay=0.03)
    
    # 包装同步函数调用
    result = mock_ai_response()
    async for chunk in wrapper.simulate_streaming(result, chunk_size=2):
        print(chunk, end='', flush=True)
    
    print("\n")
    print("✅ 函数包装测试完成")


async def main():
    print("🚀 流式功能基础测试")
    print("=" * 50)
    
    try:
        # 测试基础流式包装器
        await test_streaming_wrapper()
        
        # 测试函数包装
        await test_mock_function()
        
        print("\n🎉 所有测试通过！")
        print("💡 如果你看到了文字一点点出现，说明流式功能工作正常")
        print("📝 现在可以在 cli_chat.py 中体验真正的流式对话了")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 测试中断")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        sys.exit(1)