#!/usr/bin/env python3
"""
简化版命令行 AI 助手
直接与重庆大学知识库对话
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

def print_welcome():
    """打印欢迎信息"""
    print("\n" + "🎓" * 20)
    print("   重庆大学 AI 智能助手")
    print("🎓" * 20)
    print("\n✨ 我可以帮您查询：")
    print("📋 重庆大学政策、规定、通知")
    print("🏛️ 学校历史、文化、师资")  
    print("🎯 学生成绩、课程信息")
    print("🌐 校园生活、设施服务")
    print("\n💡 输入 'q' 退出，输入问题开始对话")
    print("-" * 40)


async def simple_kb_search(query: str) -> str:
    """简化的知识库搜索"""
    try:
        from common.tools import _sync_kb_search
        # 在线程中执行同步搜索
        result = await asyncio.to_thread(_sync_kb_search, query)
        return result
    except Exception as e:
        return f"抱歉，查询出现错误：{str(e)}"


async def simple_grade_query() -> str:
    """简化的成绩查询"""
    try:
        from common.tools import grade_query
        return await grade_query()
    except Exception as e:
        return f"抱歉，成绩查询出现错误：{str(e)}"


async def main():
    """主函数"""
    print_welcome()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 请输入您的问题: ").strip()
            
            # 退出命令
            if user_input.lower() in ['q', 'quit', 'exit', '退出']:
                print("\n👋 感谢使用重庆大学 AI 助手，再见！")
                break
            
            # 空输入检查
            if not user_input:
                print("❓ 请输入您的问题...")
                continue
            
            print("\n🤖 正在查询中...")
            
            # 简单的问题分类
            if any(keyword in user_input for keyword in ['成绩', '分数', '考试', '成绩单']):
                response = await simple_grade_query()
            else:
                response = await simple_kb_search(user_input)
            
            print(f"\n🤖 AI助手: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到退出信号，再见！")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {str(e)}")
            print("请重试或输入 'q' 退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)
