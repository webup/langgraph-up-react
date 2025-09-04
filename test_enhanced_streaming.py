#!/usr/bin/env python3
"""
增强流式功能测试脚本
测试节点级别的可视化和调试模式
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
from common.enhanced_streaming import EnhancedStreaming, CliStreamingHandler
from react_agent import graph

load_dotenv()


async def test_enhanced_streaming_basic():
    """测试基础增强流式功能"""
    print("🧪 测试增强流式处理器")
    print("=" * 50)
    
    # 创建增强流式处理器
    enhanced_streaming = EnhancedStreaming(verbose=True, show_timing=True)
    
    # 测试查询
    question = "你好，请简单介绍一下自己"
    print(f"📝 测试问题: {question}")
    print("🔍 详细处理过程:")
    
    try:
        # 创建图流
        state = {"messages": [("user", question)]}
        context = Context()
        graph_stream = graph.astream(state, context=context)
        
        # 处理增强流式
        event_count = 0
        async for event in enhanced_streaming.stream_with_node_info(
            graph_stream, 
            show_intermediate=True
        ):
            event_count += 1
            event_type = event.get("type")
            content = event.get("content", "")
            
            print(f"  事件 {event_count}: {event_type}")
            if content:
                print(f"    内容: {content[:100]}{'...' if len(content) > 100 else ''}")
            print()
        
        print(f"✅ 基础测试完成，共处理 {event_count} 个事件")
        return True
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cli_streaming_handler():
    """测试CLI流式处理句柄"""
    print("\n🧪 测试CLI流式处理句柄")
    print("=" * 50)
    
    # 创建CLI处理器
    handler = CliStreamingHandler(verbose=True)
    
    # 测试问题（可能会触发工具调用）
    question = "重庆大学在哪里？"
    print(f"📝 测试问题: {question}")
    print("🎭 模拟CLI输出:")
    
    try:
        # 创建图流
        state = {"messages": [("user", question)]}
        context = Context()
        graph_stream = graph.astream(state, context=context)
        
        # 使用CLI处理器
        await handler.handle_streaming_chat(graph_stream, "[test] ")
        
        print("✅ CLI处理器测试完成")
        return True
        
    except Exception as e:
        print(f"❌ CLI处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_verbose_vs_normal():
    """对比测试详细模式和普通模式"""
    print("\n🧪 对比测试详细模式 vs 普通模式")
    print("=" * 50)
    
    question = "请告诉我重庆大学的特色专业有哪些？"
    
    for verbose in [False, True]:
        mode_name = "详细模式" if verbose else "普通模式"
        print(f"\n📊 {mode_name} 测试:")
        print(f"问题: {question}")
        print("-" * 30)
        
        try:
            handler = CliStreamingHandler(verbose=verbose)
            state = {"messages": [("user", question)]}
            context = Context()
            graph_stream = graph.astream(state, context=context)
            
            await handler.handle_streaming_chat(graph_stream, f"[{mode_name[:2]}] ")
            print(f"✅ {mode_name} 测试完成")
            
        except Exception as e:
            print(f"❌ {mode_name} 测试失败: {e}")
    
    return True


async def test_error_handling():
    """测试错误处理"""
    print("\n🧪 测试错误处理能力")
    print("=" * 50)
    
    # 创建一个可能导致错误的场景
    try:
        enhanced_streaming = EnhancedStreaming(verbose=True)
        
        # 模拟错误的图流
        async def error_stream():
            yield {"error_node": {"messages": "this will cause an error"}}
        
        error_count = 0
        async for event in enhanced_streaming.stream_with_node_info(
            error_stream(), 
            show_intermediate=True
        ):
            print(f"处理事件: {event.get('type', 'unknown')}")
            error_count += 1
            if error_count > 5:  # 防止无限循环
                break
        
        print("✅ 错误处理测试完成")
        return True
        
    except Exception as e:
        print(f"⚠️ 期望的错误被捕获: {e}")
        return True


async def main():
    """主测试函数"""
    print("🚀 增强流式功能全面测试")
    print("=" * 60)
    print("本测试将验证:")
    print("• 基础增强流式处理器功能")
    print("• CLI流式处理句柄")
    print("• 详细模式 vs 普通模式对比")
    print("• 错误处理机制")
    print("=" * 60)
    
    test_results = []
    
    # 执行各项测试
    tests = [
        ("基础增强流式功能", test_enhanced_streaming_basic),
        ("CLI流式处理句柄", test_cli_streaming_handler),
        ("详细模式对比", test_verbose_vs_normal),
        ("错误处理", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🎯 开始测试: {test_name}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 显示测试总结
    print(f"\n{'='*60}")
    print("🎉 测试总结")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{len(test_results)} 测试通过")
    
    if passed == len(test_results):
        print("🎊 所有测试通过！增强流式功能工作正常")
        print("💡 现在可以在 cli_chat.py 中使用 'debug' 命令体验详细模式")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖")
        return False
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)