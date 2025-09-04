#!/usr/bin/env python3
"""
增强流式处理器
支持显示LangGraph节点级别的执行过程
"""

import asyncio
import time
from typing import AsyncGenerator, Dict, Any, Optional
from .streaming_wrapper import StreamingWrapper


class NodeVisualizer:
    """节点可视化器"""
    
    def __init__(self, show_details: bool = True):
        self.show_details = show_details
        self.step_counter = 0
        
    def format_node_info(self, node_name: str, step: int) -> str:
        """格式化节点信息"""
        node_icons = {
            "call_model": "🧠",
            "tools": "🔧", 
            "__start__": "🚀",
            "__end__": "✅"
        }
        
        icon = node_icons.get(node_name, "⚙️")
        return f"{icon} 步骤 {step}: {node_name}"
    
    def format_thinking(self, content: str, max_length: int = 100) -> str:
        """格式化思考内容"""
        if len(content) <= max_length:
            return f"💭 思考: {content}"
        else:
            return f"💭 思考: {content[:max_length]}..."
    
    def format_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """格式化工具调用"""
        name = tool_call.get('name', 'unknown')
        args = tool_call.get('args', {})
        
        # 简化参数显示
        if len(str(args)) > 100:
            args_str = f"{str(args)[:97]}..."
        else:
            args_str = str(args)
            
        return f"🔧 调用工具: {name}\n     参数: {args_str}"
    
    def format_tool_result(self, name: str, content: str, max_length: int = 200) -> str:
        """格式化工具结果"""
        if len(content) <= max_length:
            return f"📊 工具 '{name}' 结果: {content}"
        else:
            return f"📊 工具 '{name}' 结果: {content[:max_length]}..."


class EnhancedStreaming:
    """增强流式处理器"""
    
    def __init__(self, verbose: bool = False, show_timing: bool = False):
        self.verbose = verbose
        self.show_timing = show_timing
        self.visualizer = NodeVisualizer(show_details=verbose)
        self.streaming_wrapper = StreamingWrapper(base_delay=0.02, punct_delay=0.08)
        
    async def stream_with_node_info(
        self,
        graph_stream: AsyncGenerator[Dict[str, Any], None],
        show_intermediate: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        增强的流式处理，显示节点信息
        
        Args:
            graph_stream: LangGraph的astream输出
            show_intermediate: 是否显示中间步骤
            
        Yields:
            Dict包含: type, content, node_name, step等信息
        """
        step = 0
        start_time = time.time() if self.show_timing else None
        
        async for chunk in graph_stream:
            step += 1
            
            for node_name, node_output in chunk.items():
                # 发送节点开始信息
                if show_intermediate:
                    yield {
                        "type": "node_start",
                        "node_name": node_name,
                        "step": step,
                        "content": self.visualizer.format_node_info(node_name, step)
                    }
                
                # 处理消息
                if "messages" in node_output:
                    async for message_event in self._process_messages(
                        node_output["messages"], 
                        node_name, 
                        step, 
                        show_intermediate
                    ):
                        yield message_event
                    
                # 如果是最终的call_model节点，返回流式文本
                if node_name == "call_model" and "messages" in node_output:
                    final_message = node_output["messages"][-1]
                    if hasattr(final_message, 'content') and final_message.content:
                        # 只有当没有工具调用时才流式输出最终回答
                        if not (hasattr(final_message, 'tool_calls') and final_message.tool_calls):
                            yield {
                                "type": "final_response_start",
                                "content": ""
                            }
                            
                            # 流式输出最终回答
                            async for text_chunk in self.streaming_wrapper.simulate_streaming(
                                final_message.content, chunk_size=2
                            ):
                                yield {
                                    "type": "final_response_chunk", 
                                    "content": text_chunk
                                }
                            
                            yield {
                                "type": "final_response_end",
                                "content": ""
                            }
        
        # 发送完成信息
        if self.show_timing and start_time:
            duration = time.time() - start_time
            yield {
                "type": "completion", 
                "content": f"⏱️ 总耗时: {duration:.2f}秒"
            }
    
    async def _process_messages(
        self, 
        messages: list, 
        node_name: str, 
        step: int, 
        show_intermediate: bool
    ):
        """处理消息列表"""
        for message in messages:
            # AI思考内容
            if hasattr(message, 'content') and message.content and show_intermediate:
                # 对于中间步骤的思考，不进行流式显示，直接显示
                yield {
                    "type": "thinking",
                    "content": self.visualizer.format_thinking(message.content),
                    "node_name": node_name,
                    "step": step
                }
            
            # 工具调用
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if show_intermediate:
                        yield {
                            "type": "tool_call",
                            "content": self.visualizer.format_tool_call(tool_call),
                            "node_name": node_name,
                            "step": step,
                            "tool_name": tool_call.get('name', 'unknown')
                        }
            
            # 工具结果
            if hasattr(message, 'name') and show_intermediate:  # ToolMessage
                yield {
                    "type": "tool_result",
                    "content": self.visualizer.format_tool_result(
                        message.name, 
                        str(message.content)
                    ),
                    "node_name": node_name,
                    "step": step,
                    "tool_name": message.name
                }


class CliStreamingHandler:
    """CLI流式处理句柄"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.enhanced_streaming = EnhancedStreaming(verbose=verbose, show_timing=True)
    
    async def handle_streaming_chat(
        self,
        graph_stream: AsyncGenerator[Dict[str, Any], None],
        session_prompt: str = ""
    ):
        """
        处理CLI流式聊天
        
        Args:
            graph_stream: LangGraph的stream输出
            session_prompt: 会话提示符前缀
        """
        print("🤔 AI正在分析和处理...", end="", flush=True)
        await asyncio.sleep(0.3)
        
        final_response_started = False
        
        async for event in self.enhanced_streaming.stream_with_node_info(
            graph_stream, 
            show_intermediate=self.verbose
        ):
            event_type = event.get("type")
            content = event.get("content", "")
            
            if event_type == "node_start" and self.verbose:
                print(f"\r{' ' * 50}\r", end="")  # 清除之前的内容
                print(f"  {content}")
                
            elif event_type == "thinking" and self.verbose:
                print(f"    {content}")
                
            elif event_type == "tool_call" and self.verbose:
                print(f"    {content}")
                
            elif event_type == "tool_result" and self.verbose:
                print(f"    {content}")
                
            elif event_type == "final_response_start":
                print(f"\r{' ' * 50}\r", end="")  # 清除处理提示
                print(f"{session_prompt}🤖 AI: ", end="", flush=True)
                final_response_started = True
                
            elif event_type == "final_response_chunk" and final_response_started:
                print(content, end="", flush=True)
                
            elif event_type == "final_response_end":
                print()  # 换行
                final_response_started = False
                
            elif event_type == "completion":
                if self.verbose:
                    print(f"\n💫 {content}")


# 便捷函数
async def create_enhanced_stream(graph, state, context, verbose: bool = False):
    """创建增强流式处理"""
    handler = CliStreamingHandler(verbose=verbose)
    graph_stream = graph.astream(state, context=context)
    return handler.handle_streaming_chat(graph_stream)