import concurrent.futures
import os
import sys
import threading
import asyncio
import time
from typing import List, Dict, Any, Optional, Set
from functools import lru_cache
import hashlib

from .config import RAGFLOW, RERANK_MODEL
from .llm_server import Rerank_LLM
from .ragflow import RAGFlowRetrieval


class KB_Retrieval:
    def __init__(self, similarity_score: float = 0.5, top_k: int = 5, max_workers: int = 4, enable_cache: bool = True):
        self.rag_client = RAGFlowRetrieval(api_key=RAGFLOW.API_KEY, base_url=RAGFLOW.BASE_URL)
        self.similarity_score = similarity_score
        self.top_k = top_k
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        
        # 优化：使用集合进行去重，提高查找效率
        self.chunk_content_set: Set[str] = set()
        self.chunk_content: List[Dict[str, Any]] = []
        
        self.rerank_client = Rerank_LLM(key=RERANK_MODEL.API_KEY, model_name=RERANK_MODEL.MODEL_NAME, base_url=RERANK_MODEL.BASE_URL)
        self._lock = threading.Lock()  # 线程安全锁
        
        # 缓存机制
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 300  # 5分钟缓存时间

    def _generate_cache_key(self, questions: List[str]) -> str:
        """生成缓存键"""
        content = "|".join(sorted(questions)) + f"|{self.similarity_score}|{self.top_k}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """从缓存获取结果"""
        if not self.enable_cache or cache_key not in self._cache:
            return None
        
        # 检查缓存是否过期
        if time.time() - self._cache_timestamps[cache_key] > self._cache_ttl:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None
        
        return self._cache[cache_key]
    
    def _set_cache(self, cache_key: str, result: str) -> None:
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

    async def async_retrieve(self, question: List[str]) -> str:
        """异步版本的检索方法"""
        # 在线程池中执行同步检索
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, question)

    def retrieve(self, question: List[str]) -> str:
        # 检查缓存
        cache_key = self._generate_cache_key(question)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 清空之前的结果
        self.chunk_content_set.clear()
        self.chunk_content.clear()
        
        # 批量检索
        results = self.rag_client.batch_retrieve(
            question, 
            dataset_ids=[RAGFLOW.DATASET_ID], 
            similarity_threshold=self.similarity_score, 
            top_k=self.top_k * 2  # 获取更多候选项用于rerank
        )
        
        # 优化：使用集合快速去重
        for result in results:
            chunks = result.get("chunks", [])
            for chunk in chunks:
                content = chunk.get("content")
                if content and content not in self.chunk_content_set:
                    self.chunk_content_set.add(content)
                    self.chunk_content.append(chunk)
        
        # 如果chunks数量少于等于top_k，直接返回，跳过rerank以提升速度
        if len(self.chunk_content) <= self.top_k:
            context = ""
            for chunk in self.chunk_content:
                context += f"<chunk>\n{chunk['content']}\n</chunk>\n"
            self._set_cache(cache_key, context)
            return context
        
        # 只对有内容的chunks进行rerank
        chunk_texts = [chunk.get("content", "") for chunk in self.chunk_content]
        if not chunk_texts:
            return ""
        
        # 使用原始的rerank提示词模板进行重排序
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        document_template = "<Document>: {doc}{suffix}"
        
        query_for_rerank = question[-1] if question else ""
        query = query_template.format(prefix=prefix, instruction=instruction, query=query_for_rerank)
        texts = [document_template.format(doc=text, suffix=suffix) for text in chunk_texts]
        
        try:
            rank = self.rerank_client.similarity(query, texts)
            # 根据rerank结果排序并取top_k
            sorted_chunks = [chunk for _, chunk in sorted(zip(rank, self.chunk_content), 
                           key=lambda x: x[0], reverse=True)]
            selected_chunks = sorted_chunks[:self.top_k]
        except Exception as e:
            print(f"Rerank失败，使用原始顺序: {e}")
            selected_chunks = self.chunk_content[:self.top_k]
        
        # 构建上下文
        context = ""
        for chunk in selected_chunks:
            context += f"<chunk>\n{chunk['content']}\n</chunk>\n"
        
        # 缓存结果
        self._set_cache(cache_key, context)
        return context

    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()


if __name__ == "__main__":
    from llm_server import LLM
    llm = LLM()
    rewrite_result = llm.query_rewrite("降转政策")
    query_list = list(rewrite_result.values())[:-1]
    rag = KB_Retrieval()
    context = rag.retrieve(query_list)
    result = llm.chat_completion(query_list[-1], context)
    print(result)