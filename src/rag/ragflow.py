#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow 知识库召回接口的完整Python示例
包含Python SDK和HTTP API两种调用方式
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from ragflow_sdk import RAGFlow


class RAGFlowRetrieval:
    """RAGFlow召回接口封装类"""
    
    def __init__(self, api_key: str, base_url: str):
        """
        初始化RAGFlow召回客户端
        
        Args:
            api_key: RAGFlow API密钥
            base_url: RAGFlow服务器地址，格式如 "http://localhost:9380"
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        # 初始化Python SDK客户端
        self.rag_client = RAGFlow(api_key=api_key, base_url=base_url)
    
    def retrieve_chunks_http_api(
        self,
        question: str,
        dataset_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 30,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        top_k: int = 1024,
        rerank_id: Optional[str] = None,
        keyword: bool = False,
        highlight: bool = False
    ) -> Dict[str, Any]:
        """
        使用HTTP API调用召回接口
        
        Args:
            question: 查询问题
            dataset_ids: 数据集ID列表
            document_ids: 文档ID列表
            page: 页码
            page_size: 每页大小
            similarity_threshold: 相似度阈值
            vector_similarity_weight: 向量相似度权重
            top_k: 候选chunk数量
            rerank_id: 重排模型ID
            keyword: 是否启用关键词匹配
            highlight: 是否高亮匹配词
            
        Returns:
            召回结果字典
        """
        url = f"{self.base_url}/api/v1/retrieval"
        
        # 构建请求数据
        data = {
            "question": question,
            "page": page,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
            "keyword": keyword,
            "highlight": highlight
        }
        
        # 添加可选参数
        if dataset_ids:
            data["dataset_ids"] = dataset_ids
        if document_ids:
            data["document_ids"] = document_ids
        if rerank_id:
            data["rerank_id"] = rerank_id
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"HTTP请求错误: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {"error": str(e)}
    
    def retrieve_chunks_advanced(
        self,
        question: str,
        dataset_ids: List[str],
        similarity_threshold: float = 0.2,
        top_k: int = 10,
        enable_rerank: bool = False,
        enable_keyword: bool = False,
        enable_highlight: bool = False
    ) -> Dict[str, Any]:
        """
        高级召回接口（推荐使用）
        
        Args:
            question: 查询问题
            dataset_ids: 知识库ID列表
            similarity_threshold: 相似度阈值（0-1）
            top_k: 返回的chunk数量
            enable_rerank: 是否启用重排
            enable_keyword: 是否启用关键词匹配
            enable_highlight: 是否启用高亮显示
            
        Returns:
            格式化的召回结果
        """
        result = self.retrieve_chunks_http_api(
            question=question,
            dataset_ids=dataset_ids,
            page_size=top_k,
            similarity_threshold=similarity_threshold,
            keyword=enable_keyword,
            highlight=enable_highlight
        )
        
        if "error" in result:
            return result
        
        # 格式化返回结果
        if result.get("code") == 0 and "data" in result:
            data = result["data"]
            formatted_result = {
                "question": question,
                "total_chunks": data.get("total", 0),
                "chunks": [],
                "document_stats": data.get("doc_aggs", [])
            }
            
            # 格式化chunk信息
            for chunk in data.get("chunks", []):
                formatted_chunk = {
                    "id": chunk.get("id"),
                    "content": chunk.get("content"),
                    "document_name": chunk.get("document_keyword"),
                    "document_id": chunk.get("document_id"),
                    "similarity_score": chunk.get("similarity", 0),
                    "vector_similarity": chunk.get("vector_similarity", 0),
                    "term_similarity": chunk.get("term_similarity", 0),
                    "highlighted_content": chunk.get("highlight", ""),
                    "important_keywords": chunk.get("important_keywords", [])
                }
                formatted_result["chunks"].append(formatted_chunk)
            
            return formatted_result
        else:
            return {"error": result.get("message", "Unknown error")}
    
    def batch_retrieve(
        self,
        questions: List[str],
        dataset_ids: List[str],
        similarity_threshold: float = 0.2,
        top_k: int = 5,
        max_workers: int = None
    ) -> List[Dict[str, Any]]:
        """
        批量召回查询（多线程版本）
        
        Args:
            questions: 问题列表
            dataset_ids: 知识库ID列表
            similarity_threshold: 相似度阈值
            top_k: 每个问题返回的chunk数量
            max_workers: 最大线程数，默认为None（系统自动选择）
            
        Returns:
            批量召回结果列表
        """
        if not questions:
            return []
        
        # 如果只有一个问题，直接调用单个查询
        if len(questions) == 1:
            result = self.retrieve_chunks_advanced(
                question=questions[0],
                dataset_ids=dataset_ids,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
            return [result]
        
        # 使用ThreadPoolExecutor进行多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_question = {
                executor.submit(
                    self.retrieve_chunks_advanced,
                    question=question,
                    dataset_ids=dataset_ids,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k
                ): question
                for question in questions
            }
            
            # 收集结果，保持原始顺序
            results = [None] * len(questions)
            question_to_index = {question: i for i, question in enumerate(questions)}
            
            for future in as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    # 将结果放在正确的位置以保持原始顺序
                    index = question_to_index[question]
                    results[index] = result
                except Exception as exc:
                    print(f'问题 "{question}" 处理时发生异常: {exc}')
                    # 在发生异常时创建一个错误结果
                    index = question_to_index[question]
                    results[index] = {"error": f"处理异常: {exc}", "question": question}
        
        return results
    
    def search_in_specific_documents(
        self,
        question: str,
        document_ids: List[str],
        similarity_threshold: float = 0.1,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        在指定文档中搜索
        
        Args:
            question: 查询问题
            document_ids: 文档ID列表
            similarity_threshold: 相似度阈值
            top_k: 返回数量
            
        Returns:
            搜索结果
        """
        return self.retrieve_chunks_http_api(
            question=question,
            document_ids=document_ids,
            page_size=top_k,
            similarity_threshold=similarity_threshold,
            keyword=True,
            highlight=True
        )
    
    def print_retrieval_results(self, results: Dict[str, Any]):
        """
        打印召回结果
        
        Args:
            results: 召回结果字典
        """
        if "error" in results:
            print(f"❌ 错误: {results['error']}")
            return
        
        print(f"🔍 查询问题: {results.get('question', 'N/A')}")
        print(f"📊 总共找到 {results.get('total_chunks', 0)} 个相关chunk")
        print(f"📄 涉及文档: {len(results.get('document_stats', []))} 个")
        
        print("\n" + "="*60)
        print("📋 召回结果详情:")
        print("="*60)
        
        for i, chunk in enumerate(results.get('chunks', []), 1):
            print(f"\n🔹 Chunk {i}:")
            print(f"   ID: {chunk.get('id', 'N/A')}")
            print(f"   文档: {chunk.get('document_name', 'N/A')}")
            print(f"   相似度: {chunk.get('similarity_score', 0):.4f}")
            print(f"   向量相似度: {chunk.get('vector_similarity', 0):.4f}")
            print(f"   词汇相似度: {chunk.get('term_similarity', 0):.4f}")
            
            content = chunk.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   内容: {content}")
            
            if chunk.get('highlighted_content'):
                highlighted = chunk.get('highlighted_content', '')
                if len(highlighted) > 200:
                    highlighted = highlighted[:200] + "..."
                print(f"   高亮: {highlighted}")
            
            if chunk.get('important_keywords'):
                print(f"   关键词: {', '.join(chunk.get('important_keywords', []))}")
        
        print("\n" + "="*60)
        print("📈 文档统计:")
        print("="*60)
        for doc_stat in results.get('document_stats', []):
            print(f"   📄 {doc_stat.get('doc_name', 'N/A')}: {doc_stat.get('count', 0)} 个chunk")

    def find_dataset_id(self) -> str:
        # 获取知识库列表
        try:
            datasets = self.rag_client.list_datasets()
            if not datasets:
                print("❌ 没有找到任何知识库，请先创建知识库并上传文档")
                return None
            
            print(f"📚 找到 {len(datasets)} 个知识库:")
            for i, dataset in enumerate(datasets, 1):
                print(f"   {i}. {getattr(dataset, 'name', 'N/A')} (ID: {getattr(dataset, 'id', 'N/A')})")
            
            # 使用第一个知识库进行测试
            dataset_id = getattr(datasets[0], 'id', '')
            dataset_name = getattr(datasets[0], 'name', 'N/A')
            
            if not dataset_id:
                print("❌ 无法获取知识库ID")
                return None
            
            print(f"\n🎯 使用知识库: {dataset_name} (ID: {dataset_id})")
            
        except Exception as e:
            print(f"❌ 获取知识库失败: {e}")
            return None


def main():
    """主函数 - 使用示例"""
    
    # 配置RAGFlow信息
    API_KEY = "ragflow-FiOTQ1MGI0MWMyNDExZjA4N2VlMDI0Mm"  # 替换为你的API密钥
    BASE_URL = "http://172.18.81.4:8080"  # 替换为你的RAGFlow服务器地址
    DATASET_ID = "01509e3e62cd11f0b66d0242ac140006"
    
    # 创建召回客户端
    retrieval = RAGFlowRetrieval(api_key=API_KEY, base_url=BASE_URL)
    
    # 示例1: 基础召回查询
    print("📝 示例: 基础召回查询")
    print("="*60)
    
    question1 = "开学时间"  # 替换为你想查询的问题
    results1 = retrieval.retrieve_chunks_advanced(
        question=question1,
        dataset_ids=[DATASET_ID],
        similarity_threshold=0.5,
        top_k=6,
        enable_keyword=False,
        enable_highlight=False
    )
    
    retrieval.print_retrieval_results(results1)
    
    # 示例2: 批量召回查询（多线程版本）
    print("\n\n📝 示例: 批量召回查询（多线程版本）")
    print("="*60)
    
    questions = [
        "校内vpn",
        "如何连接校园网",
        "图书馆开放时间",
        "学生宿舍管理规定",
        "课程选课系统"
    ]
    
    print(f"🚀 正在并发处理 {len(questions)} 个问题...")
    start_time = time.time()
    
    batch_results = retrieval.batch_retrieve(
        questions=questions,
        dataset_ids=[DATASET_ID],
        similarity_threshold=0.1,
        top_k=3,
        max_workers=3  # 设置最大3个线程
    )
    
    end_time = time.time()
    print(f"⏱️ 批量查询完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 处理了 {len(batch_results)} 个查询结果")
    
    # 打印每个查询的结果摘要
    for i, result in enumerate(batch_results, 1):
        if "error" in result:
            print(f"❌ 查询 {i}: {result['error']}")
        else:
            chunks_count = len(result.get('chunks', []))
            question = result.get('question', questions[i-1])
            print(f"✅ 查询 {i} ('{question}'): 找到 {chunks_count} 个相关chunk")


if __name__ == "__main__":
    main()