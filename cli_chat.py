#!/usr/bin/env python3
"""
命令行交互式 AI 助手
支持多轮对话、会话管理和知识库查询
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# 添加 src 目录到 Python 路径
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

from common.context import Context
from common.conversation_manager import (
    ChatInterface,
    ConversationManager,
    FileStorage,
    HistoryManager,
)

load_dotenv()


class CommandLineChat:
    """命令行聊天接口，支持会话管理"""

    def __init__(self):
        # 使用文件存储以保持会话持久化
        storage = FileStorage("./conversations")
        history_manager = HistoryManager(max_messages=100, max_tokens=8000)
        conversation_manager = ConversationManager(
            storage=storage, history_manager=history_manager, auto_save=True
        )

        self.chat_interface = ChatInterface(
            conversation_manager=conversation_manager, default_context=Context()
        )

        self.current_session_id: Optional[str] = None
        self.session_name: Optional[str] = None

    async def start_chat(self):
        """开始命令行对话"""
        print("🤖 重庆大学 AI 助手 (支持会话管理)")
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
        print("  - 输入 'new' 创建新会话")
        print("  - 输入 'sessions' 查看所有会话")
        print("  - 输入 'switch <session_id>' 切换会话")
        print("  - 输入 'delete <session_id>' 删除会话")
        print("  - 输入 'clear' 清空当前会话")
        print("  - 输入 'help' 查看帮助")
        print("=" * 50)

        # 创建默认会话
        await self._ensure_session()

        while True:
            try:
                # 显示会话提示符
                session_prompt = f"[{self._get_session_display()}] "
                user_input = input(f"\n{session_prompt}👤 您: ").strip()

                # 处理特殊命令
                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("\n👋 再见！感谢使用重庆大学 AI 助手！")
                    break

                elif user_input.lower() in ["new", "新建"]:
                    await self._create_new_session()
                    continue

                elif user_input.lower() == "sessions":
                    await self._list_sessions()
                    continue

                elif user_input.lower().startswith("switch "):
                    session_id = user_input[7:].strip()
                    await self._switch_session(session_id)
                    continue

                elif user_input.lower().startswith("delete "):
                    session_id = user_input[7:].strip()
                    await self._delete_session(session_id)
                    continue

                elif user_input.lower() in ["clear", "清空"]:
                    await self._clear_current_session()
                    continue

                elif user_input.lower() in ["help", "帮助"]:
                    self.show_help()
                    continue

                elif not user_input:
                    print("❓ 请输入您的问题...")
                    continue

                # 确保有活跃会话
                await self._ensure_session()

                # 调用 AI 助手
                print(f"\n{session_prompt}🤖 AI: ", end="", flush=True)
                response = await self.chat_interface.chat(
                    user_input, session_id=self.current_session_id
                )
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 检测到 Ctrl+C，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 出现错误: {str(e)}")
                print("请稍后重试或联系管理员")

    async def _ensure_session(self):
        """确保有活跃的会话"""
        if self.current_session_id is None:
            self.current_session_id = await self.chat_interface.start_conversation()
            self.session_name = "默认会话"
            print(f"✅ 已创建新会话: {self._get_session_display()}")

    async def _create_new_session(self):
        """创建新会话"""
        self.current_session_id = await self.chat_interface.start_conversation()
        self.session_name = f"会话-{self.current_session_id[:8]}"
        print(f"✅ 已创建新会话: {self._get_session_display()}")

    async def _list_sessions(self):
        """列出所有会话"""
        sessions = await self.chat_interface.list_conversations()
        if not sessions:
            print("📝 暂无保存的会话")
            return

        print("\n📝 所有会话：")
        for i, session_id in enumerate(sessions, 1):
            status = "当前" if session_id == self.current_session_id else ""
            print(f"  {i}. {session_id[:8]}...{session_id[-8:]} {status}")
        print(f"总计: {len(sessions)} 个会话")

    async def _switch_session(self, session_id: str):
        """切换会话"""
        # 支持短ID匹配
        sessions = await self.chat_interface.list_conversations()
        matched_session = None

        for s in sessions:
            if s == session_id or s.startswith(session_id):
                matched_session = s
                break

        if matched_session:
            self.current_session_id = matched_session
            self.session_name = f"会话-{matched_session[:8]}"
            print(f"✅ 已切换到会话: {self._get_session_display()}")
        else:
            print(f"❌ 找不到会话: {session_id}")

    async def _delete_session(self, session_id: str):
        """删除会话"""
        # 支持短ID匹配
        sessions = await self.chat_interface.list_conversations()
        matched_session = None

        for s in sessions:
            if s == session_id or s.startswith(session_id):
                matched_session = s
                break

        if matched_session:
            if matched_session == self.current_session_id:
                print("❌ 无法删除当前活跃会话，请先切换到其他会话")
                return

            success = await self.chat_interface.clear_conversation(matched_session)
            if success:
                print(f"✅ 已删除会话: {matched_session[:8]}")
            else:
                print(f"❌ 删除会话失败: {matched_session[:8]}")
        else:
            print(f"❌ 找不到会话: {session_id}")

    async def _clear_current_session(self):
        """清空当前会话"""
        if self.current_session_id:
            # 删除当前会话并创建新会话
            await self.chat_interface.clear_conversation(self.current_session_id)
            self.current_session_id = await self.chat_interface.start_conversation()
            self.session_name = f"会话-{self.current_session_id[:8]}"
            print(f"✅ 已清空当前会话，创建新会话: {self._get_session_display()}")
        else:
            print("❌ 没有活跃的会话需要清空")

    def _get_session_display(self) -> str:
        """获取会话显示名称"""
        if self.current_session_id:
            return f"{self.current_session_id[:8]}"
        return "无会话"

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
        print("⚙️ 基础命令：")
        print("  • quit/exit/退出 - 退出程序")
        print("  • help/帮助 - 显示此帮助信息")
        print()
        print("📝 会话管理命令：")
        print("  • new/新建 - 创建新会话")
        print("  • sessions - 查看所有会话")
        print("  • switch <session_id> - 切换到指定会话")
        print("  • delete <session_id> - 删除指定会话")
        print("  • clear/清空 - 清空当前会话")
        print()
        print("💡 会话功能：")
        print("  • 自动保存对话历史到文件")
        print("  • 支持多个独立会话")
        print("  • 智能历史压缩，防止上下文过长")
        print("  • 会话ID支持前缀匹配")
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
