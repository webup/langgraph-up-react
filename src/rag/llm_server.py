import json
import os
import sys
import time
import hashlib
from typing import Dict, List, Optional, Any
from functools import lru_cache

import asyncio
import numpy as np
import aiohttp
import requests
from .config import LLM as LLM_CONFIG
from openai import AsyncOpenAI, OpenAI


# 包装为类
class LLM:
    def __init__(self, enable_cache: bool = True):
        self.client = OpenAI(
            api_key=LLM_CONFIG.API_KEY,
            base_url=LLM_CONFIG.BASE_URL,
            timeout=30,  # 设置超时
            max_retries=2  # 设置重试次数
        )
        self.async_client = AsyncOpenAI(
            api_key=LLM_CONFIG.API_KEY,
            base_url=LLM_CONFIG.BASE_URL,
            timeout=30,
            max_retries=2
        )
        
        # 缓存机制
        self.enable_cache = enable_cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300  # 5分钟缓存
    
    def _generate_cache_key(self, method: str, *args) -> str:
        """生成缓存键"""
        content = f"{method}|" + "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def _cached_query_rewrite(self, query: str) -> Dict[str, Any]:
        """带LRU缓存的查询重写"""
        return self._uncached_query_rewrite(query)
    
    def _uncached_query_rewrite(self, query: str) -> Dict[str, Any]:
        """不缓存的查询重写实现"""
        response = self.client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            extra_body={"enable_thinking": False},
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': LLM_CONFIG.REWRITE_PROMPT.format(user_input=query)}
            ]
        )
        return json.loads(response.choices[0].message.content)

    def query_rewrite(self, query: str) -> Dict[str, Any]:
        """查询重写（带缓存）"""
        if not query or not query.strip():
            return {"error": "查询不能为空"}
            
        if self.enable_cache:
            return self._cached_query_rewrite(query.strip())
        else:
            return self._uncached_query_rewrite(query.strip())

    async def async_query_rewrite(self, query: str):
        response = await self.async_client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            extra_body={"enable_thinking": False},
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': LLM_CONFIG.REWRITE_PROMPT.format(user_input=query)}
            ]
        )
        return json.loads(response.choices[0].message.content)

    def chat_completion(self, query: str, context: str):
        response = self.client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': LLM_CONFIG.CHAT_PROMPT.format(user_input=query, context=context)}
            ]
        )
        return response.choices[0].message.content

    async def async_chat_completion(self, query: str, context: str):
        response = await self.async_client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': LLM_CONFIG.CHAT_PROMPT.format(user_input=query, context=context)}
            ]
        )
        return response.choices[0].message.content

    def chat_completion_with_history(self, memory: list, history: list, messages: list):
        response = self.client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': "以下内容是记忆信息" + str(memory)},
                {'role': 'user', 'content': "以下内容是用户和AI的对话历史" + str(history)}
            ] + messages
        )
        return response.choices[0].message.content

    def memory_completion(self, query: str):
        response = self.client.chat.completions.create(
            model=LLM_CONFIG.MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': LLM_CONFIG.MEMORY_PROMPT.replace("{data}", query)}
            ]
        )
        return response.choices[0].message.content

class Rerank_LLM():
    def __init__(self, key, model_name, base_url=None, enable_cache: bool = True):
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/') if base_url else None
        self.enable_cache = enable_cache
        
        # 优化：配置session以提高性能
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        })
        # 连接池优化
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=1
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 缓存机制
        self._cache: Dict[str, List[float]] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 600  # 10分钟缓存

    def _generate_cache_key(self, query: str, texts: List[str]) -> str:
        """生成缓存键"""
        content = query + "|" + "|".join(texts)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """从缓存获取结果"""
        if not self.enable_cache or cache_key not in self._cache:
            return None
        
        # 检查缓存是否过期
        if time.time() - self._cache_timestamps[cache_key] > self._cache_ttl:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._cache[cache_key]
    
    def _set_cache(self, cache_key: str, result: List[float]) -> None:
        """设置缓存"""
        if not self.enable_cache:
            return
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
        
        # 清理过期缓存
        current_time = time.time()
        expired_keys = [k for k, t in self._cache_timestamps.items() 
                       if current_time - t > self._cache_ttl]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

    def similarity(self, query: str, texts: List[str]) -> List[float]:
        """计算相似度（带缓存）"""
        if not texts or not query.strip():
            return [0.0] * len(texts)
        
        # 检查缓存
        cache_key = self._generate_cache_key(query, texts)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 计算相似度
        result = self._sync_similarity_in_thread(query, texts)
        
        # 缓存结果
        self._set_cache(cache_key, result)
        return result

    async def async_similarity(self, query: str, texts: List[str]) -> List[float]:
        """异步计算相似度"""
        return await asyncio.to_thread(self.similarity, query, texts)

    def _sync_similarity_in_thread(self, query: str, texts: List[str]) -> List[float]:
        """同步计算相似度的核心实现"""
        try:
            if not texts:
                return []
            
            # 优化：对于少量文本，使用简单排序避免API调用
            if len(texts) <= 3:
                # 简单的文本匹配评分
                scores = []
                query_lower = query.lower()
                for text in texts:
                    text_lower = text.lower()
                    # 计算简单的重叠度
                    common_words = set(query_lower.split()) & set(text_lower.split())
                    score = len(common_words) / max(len(query_lower.split()), 1)
                    scores.append(score)
                return scores
            
            # 对于大量文本，使用rerank API
            url = f"{self.base_url}/rerank"
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": texts,
                "top_n": len(texts),
                "return_documents": False
            }
            
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            # 解析结果
            rank = np.zeros(len(texts), dtype=float)
            if 'results' in result:
                for item in result['results']:
                    if 'index' in item and 'relevance_score' in item:
                        idx = item['index']
                        if 0 <= idx < len(texts):
                            rank[idx] = float(item['relevance_score'])
            
            return rank.tolist()
            
        except requests.exceptions.Timeout:
            print("Rerank timeout, using fallback scoring")
            return self._fallback_scoring(query, texts)
        except Exception as e:
            print(f"Rerank error: {e}, using fallback scoring")
            return self._fallback_scoring(query, texts)
    
    def _fallback_scoring(self, query: str, texts: List[str]) -> List[float]:
        """回退评分方法"""
        scores = []
        
        # 提取查询中的实际问题部分（跳过格式化标记）
        if "<Query>:" in query:
            query_parts = query.split("<Query>:")
            if len(query_parts) > 1:
                actual_query = query_parts[1].split("\n")[0].strip()
            else:
                actual_query = query
        else:
            actual_query = query
        
        query_lower = actual_query.lower()
        query_words = set(query_lower.split())
        
        for text in texts:
            # 提取文档中的实际内容部分（跳过格式化标记）
            if "<Document>:" in text:
                text_parts = text.split("<Document>:")
                if len(text_parts) > 1:
                    actual_text = text_parts[1].split("<|im_end|>")[0].strip()
                else:
                    actual_text = text
            else:
                actual_text = text
                
            text_lower = actual_text.lower()
            text_words = set(text_lower.split())
            
            # 简单的Jaccard相似度
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            
            if union == 0:
                score = 0.0
            else:
                score = intersection / union
                
            scores.append(score)
        
        return scores
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
        
if __name__ == "__main__":
    llm = LLM()
    print(llm.query_rewrite("降转政策"))      