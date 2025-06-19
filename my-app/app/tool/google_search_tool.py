from typing import List

from langchain_core.tools import tool
import http.client
import json


@tool
def google_search(query: str) -> List:
    """
     从google查询答案
    :param query:  问题
    :return:  回答数组 {"title":"","link":"","snippet":"","position":""}:
    """
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": f"{query}"
    })
    headers = {
        'X-API-KEY': '097275c969319c37006768fe1ececcd4a1af1c63',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    json_message = json.loads(data.decode("utf-8"))

    results = []

    for item in json_message.get('organic', []):
        result = {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "position": item.get("position")
        }
        results.append(result)

    return results