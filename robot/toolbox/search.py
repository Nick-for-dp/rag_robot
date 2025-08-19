from typing import List
from langchain_core.tools import tool
from langchain_community.retrievers import TavilySearchAPIRetriever
from robot.utils import get_api_key


@tool(parse_docstring=True)
def tavily_search(query: str) -> List[str]:
    """使用 Tavily API 执行网络搜索并返回相关结果。

    Args:
        query (str): 搜索查询字符串

    Returns:
        List[str]: 包含搜索结果的文档内容

    Raises:
        ValueError: 如果 TAVILY_API_KEY 环境变量未设置
        Exception: Tavily API 调用失败时抛出
    """
    tavily_api_key = get_api_key("TAVILY_API_KEY")
    retriever = TavilySearchAPIRetriever(k=3, api_key=tavily_api_key)
    results = [doc.page_content for doc in retriever.invoke(query)]
    return results
