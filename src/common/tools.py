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


async def get_tools() -> List[Callable[..., Any]]:
    """Get all available tools based on configuration."""
    tools = [web_search, grade_query]

    runtime = get_runtime(Context)

    if runtime.context.enable_deepwiki:
        deepwiki_tools = await get_deepwiki_tools()
        tools.extend(deepwiki_tools)
        logger.info(f"Loaded {len(deepwiki_tools)} deepwiki tools")

    return tools
