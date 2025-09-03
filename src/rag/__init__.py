"""RAG (Retrieval-Augmented Generation) Module.

This module provides comprehensive RAG functionality including:
- Knowledge base retrieval using RAGFlow
- LLM services for query rewriting and chat completion
- Reranking capabilities for improving retrieval quality

The module supports both single and batch query processing with multi-threading
for improved performance.
"""

from .llm_server import LLM, Rerank_LLM
from .rag import KB_Retrieval
from .ragflow import RAGFlowRetrieval

__all__ = [
    "LLM",
    "Rerank_LLM", 
    "KB_Retrieval",
    "RAGFlowRetrieval"
]
