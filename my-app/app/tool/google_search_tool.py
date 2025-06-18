from langchain_core.tools import tool
from serpapi import GoogleSearch


@tool
def google_search(query: str) -> str:
    """
     从google查询答案
    :param query:  问题
    :return:  回答
    """
    params = {
        "engine": "google",
        "q": f"{query}",
        "location": "Seattle-Tacoma, WA, Washington, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "num": "10",
        "start": "10",
        "safe": "active",
        "api_key": "bb0d0f771a56b66c76f5e3350e84ede23f30c1cf009a3608b7cd9c6cf0387e56"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    return organic_results[0]["snippet"]
