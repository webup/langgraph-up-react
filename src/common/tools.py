"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import asyncio
import logging
from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from common.context import Context
from common.mcp import get_deepwiki_tools, get_all_mcp_tools

logger = logging.getLogger(__name__)


async def web_search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def grade_query() -> str:
    """查询学生的成绩信息
    """
    search_result = """
    线性代数：90
    高等数学：85
    大学英语：88
    体育：92
    思想政治理论：89
    军事训练：91
    军事理论：88
    """
    return search_result.strip()

async def KB_search(query: str) -> str:
    """查询与重庆大学相关知识，包括：
    1、政策、通知、计算规则等相关知识
    2、重庆大学的历史、文化、特色、传统等相关知识
    3、重庆大学的师资力量、科研成果、学术交流等相关知识
    4、重庆大学的校园生活、学生管理、就业指导等相关知识
    5、重庆大学的校园环境、设施、服务等相关知识
    6、重庆大学的校园文化、活动、社团等相关知识
    7、重庆大学的校园新闻、公告、通知等相关知识

    Args:
        query: The user's search query
    """
    try:
        # 将整个同步操作移到线程中执行
        return await asyncio.to_thread(_sync_kb_search, query)
        
    except Exception as e:
        logger.error(f"Error in KB_search: {str(e)}")
        return f"抱歉，知识库搜索过程中出现错误：{str(e)}"


def _sync_kb_search(query: str) -> str:
    """同步版本的知识库搜索，在线程中执行"""
    try:
        from rag import LLM, KB_Retrieval
        
        # Initialize components
        llm = LLM()
        rag = KB_Retrieval()

        # Rewrite query to improve retrieval
        rewrite_result = llm.query_rewrite(query)
        
        # Extract query variants (excluding the last element which might be metadata)
        query_list = list(rewrite_result.values())[:-1]
        
        # Ensure we have valid queries
        if not query_list:
            query_list = [query]  # Fallback to original query
        
        # Retrieve relevant context from knowledge base
        context = rag.retrieve(query_list)
        
        # Generate final answer using the last query variant and retrieved context
        final_query = query_list[-1] if query_list else query
        result = llm.chat_completion(final_query, context)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in _sync_kb_search: {str(e)}")
        return f"抱歉，知识库搜索过程中出现错误：{str(e)}"


async def get_tools() -> List[Callable[..., Any]]:
    """Get all available tools based on configuration."""
    tools = [grade_query, KB_search]

    runtime = get_runtime(Context)

    if runtime.context.enable_deepwiki:
        deepwiki_tools = await get_deepwiki_tools()
        tools.extend(deepwiki_tools)
        logger.info(f"Loaded {len(deepwiki_tools)} deepwiki tools")
    
    # 获取所有工具
    all_tools = await get_all_mcp_tools()
    tools.extend(all_tools)
    logger.info(f"Loaded {len(all_tools)} all tools")

    return tools
