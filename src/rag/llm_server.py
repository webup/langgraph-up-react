import json
import os
import sys

import numpy as np
import requests
from config import LLM as LLM_CONFIG
from openai import OpenAI


# 包装为类
class LLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG.API_KEY,
            base_url=LLM_CONFIG.BASE_URL,
        )

    def query_rewrite(self, query: str):
        response = self.client.chat.completions.create(
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
    def __init__(self, key, model_name, base_url=None):
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/') if base_url else None
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        })

    def similarity(self, query: str, texts: list):
        try:
            # 尝试使用rerank专用接口
            url = f"{self.base_url}/rerank"
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": texts,
                "top_n": len(texts),
                "return_documents": False
            }
            
            response = self.session.post(url, json=payload)       
            response.raise_for_status()
            result = response.json()
            
            # 解析结果
            rank = np.zeros(len(texts), dtype=float)
            if 'results' in result:
                for item in result['results']:
                    if 'index' in item and 'relevance_score' in item:
                        rank[item['index']] = item['relevance_score']
            
            # 使用list返回
            return np.array(rank).tolist()
            
        except Exception as e:
            print(f"Rerank error: {e}")
        
if __name__ == "__main__":
    llm = LLM()
    print(llm.query_rewrite("降转政策"))      