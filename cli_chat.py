#!/usr/bin/env python3
"""
命令行交互式 AI 助手
支持多轮对话和知识库查询
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
from react_agent import graph

load_dotenv()

class CommandLineChat:
    """命令行聊天接口"""
    
    def __init__(self):
        self.context = Context()
        self.conversation_history = []
        
    async def start_chat(self):
        """开始命令行对话"""
        print("🤖 重庆大学 AI 助手")
        print("=" * 50)
        print("欢迎使用重庆大学智能助手！我可以帮您查询：")
        print("• 📚 重庆大学相关政策、通知、规定")
        print("• 🎓 学校历史、文化、师资力量")
        print("• 🏛️ 校园环境、设施、服务")
        print("• 📊 学生成绩查询")
        print("• 🌐 通用知识查询")
        print("-" * 50)
        print("💡 提示：")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'clear' 清空对话历史")
        print("  - 输入 'help' 查看帮助")
        print("=" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 再见！感谢使用重庆大学 AI 助手！")
                    break
                    
                elif user_input.lower() in ['clear', '清空']:
                    self.conversation_history = []
                    print("\n✅ 对话历史已清空")
                    continue
                    
                elif user_input.lower() in ['help', '帮助']:
                    self.show_help()
                    continue
                    
                elif not user_input:
                    print("❓ 请输入您的问题...")
                    continue
                
                # 调用 AI 助手
                print("\n🤖 AI: ", end="", flush=True)
                response = await self.get_ai_response(user_input)
                print(response)
                
                # 保存对话历史
                self.conversation_history.append({
                    "user": user_input,
                    "ai": response
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 出现错误: {str(e)}")
                print("请稍后重试或联系管理员")
    
    async def get_ai_response(self, user_input: str) -> str:
        """获取 AI 响应"""
        try:
            # 构建消息历史
            messages = []
            
            # 添加历史对话（保留最近5轮对话）
            recent_history = self.conversation_history[-5:]
            for conv in recent_history:
                messages.append(("user", conv["user"]))
                messages.append(("assistant", conv["ai"]))
            
            # 添加当前用户输入
            messages.append(("user", user_input))
            
            # 调用图
            result = await graph.ainvoke(
                {"messages": messages},
                context=Context(),
                config={"run_name": "cli_chat"}
            )
            
            # 提取最后的 AI 消息
            if result and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            
            return "抱歉，我没有理解您的问题，请重新描述一下。"
            
        except Exception as e:
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 重庆大学 AI 助手使用帮助")
        print("=" * 40)
        print("🔍 查询示例：")
        print("  • '重庆大学的转专业政策是什么？'")
        print("  • '我的数学成绩如何？'")  
        print("  • '重庆大学有哪些特色专业？'")
        print("  • '校园网如何连接？'")
        print("  • '图书馆开放时间是什么？'")
        print()
        print("⚙️ 命令说明：")
        print("  • quit/exit/退出 - 退出程序")
        print("  • clear/清空 - 清空对话历史") 
        print("  • help/帮助 - 显示此帮助信息")
        print("=" * 40)


async def main():
    """主函数"""
    chat = CommandLineChat()
    await chat.start_chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)
