#!/usr/bin/env python3
"""
流式输出包装器
为非流式接口提供真正的流式体验
"""

import asyncio
import re
from typing import AsyncGenerator, List


class StreamingWrapper:
    """流式输出包装器"""
    
    def __init__(self, base_delay: float = 0.03, punct_delay: float = 0.1):
        self.base_delay = base_delay  # 基础延迟（秒）
        self.punct_delay = punct_delay  # 标点符号后的延迟
        
    async def simulate_streaming(self, text: str, chunk_size: int = 2) -> AsyncGenerator[str, None]:
        """
        将完整文本转换为流式输出
        
        Args:
            text: 要流式输出的文本
            chunk_size: 每个chunk的字符数
        """
        if not text:
            return
            
        # 智能分块
        chunks = self._smart_split(text, chunk_size)
        
        for chunk in chunks:
            yield chunk
            
            # 动态延迟：标点符号后延迟更长
            delay = self._calculate_delay(chunk)
            await asyncio.sleep(delay)
    
    def _smart_split(self, text: str, chunk_size: int) -> List[str]:
        """智能分割文本，考虑标点符号和自然断点"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        i = 0
        
        while i < len(text):
            # 确定chunk的结束位置
            end_pos = min(i + chunk_size, len(text))
            
            # 如果不是最后一个chunk，尝试找到更好的断点
            if end_pos < len(text):
                # 在附近寻找标点符号或空格
                best_break = end_pos
                
                # 向前搜索，找到最佳断点
                for j in range(end_pos, max(i, end_pos - chunk_size // 2), -1):
                    char = text[j]
                    if char in '，。！？；：\n ':
                        best_break = j + 1
                        break
                    elif char in '"）】』」':  # 右括号类
                        best_break = j + 1
                        break
                
                end_pos = best_break
            
            chunk = text[i:end_pos]
            if chunk:
                chunks.append(chunk)
            
            i = end_pos
        
        return chunks
    
    def _calculate_delay(self, chunk: str) -> float:
        """根据文本内容计算延迟时间"""
        # 基础延迟
        delay = self.base_delay
        
        # 如果包含标点符号，增加延迟
        if re.search(r'[。！？：；]', chunk):
            delay += self.punct_delay * 2
        elif re.search(r'[，、]', chunk):
            delay += self.punct_delay
        elif re.search(r'[\n]', chunk):
            delay += self.punct_delay * 1.5
        
        # 根据chunk长度调整延迟
        delay += len(chunk) * 0.005
        
        return delay
    
    @staticmethod
    async def wrap_non_streaming_call(coro, chunk_size: int = 3) -> AsyncGenerator[str, None]:
        """
        包装非流式协程调用，提供流式输出体验
        
        Args:
            coro: 非流式的协程函数
            chunk_size: 流式输出的chunk大小
        """
        wrapper = StreamingWrapper()
        
        # 执行原始调用获取完整结果
        try:
            result = await coro
            if result and isinstance(result, str):
                # 将结果转换为流式输出
                async for chunk in wrapper.simulate_streaming(result, chunk_size):
                    yield chunk
            else:
                # 如果没有结果，返回空
                yield ""
        except Exception as e:
            # 错误情况下也要有流式体验
            error_msg = f"⚠️ 处理请求时发生错误: {str(e)}"
            async for chunk in wrapper.simulate_streaming(error_msg, chunk_size):
                yield chunk


# 便捷函数
async def stream_text(text: str, chunk_size: int = 3, 
                     base_delay: float = 0.03) -> AsyncGenerator[str, None]:
    """便捷的文本流式输出函数"""
    wrapper = StreamingWrapper(base_delay=base_delay)
    async for chunk in wrapper.simulate_streaming(text, chunk_size):
        yield chunk


async def stream_function_call(func, *args, chunk_size: int = 3, **kwargs) -> AsyncGenerator[str, None]:
    """包装函数调用为流式输出"""
    if asyncio.iscoroutinefunction(func):
        coro = func(*args, **kwargs)
    else:
        # 同步函数转异步
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, func, *args, **kwargs)
        coro = asyncio.sleep(0, result)  # 创建一个返回结果的协程
    
    async for chunk in StreamingWrapper.wrap_non_streaming_call(coro, chunk_size):
        yield chunk