"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import logging
from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from common.context import Context
from common.mcp import get_deepwiki_tools

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
    """Get student grade information.

    Returns academic grades for various subjects including mathematics, 
    English, sports, and political theory courses.
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
    """Search for knowledge base results using RAG pipeline.
    
    This function performs a multi-step search process:
    1. Rewrites the input query to generate multiple search variants
    2. Retrieves relevant document chunks from the knowledge base  
    3. Uses LLM to generate a comprehensive answer based on retrieved context
    
    Args:
        query: The user's search query
    """
    try:
        from rag.llm_server import LLM
        from rag.rag import KB_Retrieval
        
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
        logger.error(f"Error in KB_search: {str(e)}")
        return f"抱歉，知识库搜索过程中出现错误：{str(e)}"


async def get_tools() -> List[Callable[..., Any]]:
    """Get all available tools based on configuration."""
    tools = [grade_query, KB_search]

    runtime = get_runtime(Context)

    if runtime.context.enable_deepwiki:
        deepwiki_tools = await get_deepwiki_tools()
        tools.extend(deepwiki_tools)
        logger.info(f"Loaded {len(deepwiki_tools)} deepwiki tools")

    return tools
