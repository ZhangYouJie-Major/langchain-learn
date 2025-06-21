from langchain_community.utilities import (
    SerpAPIWrapper,
    WikipediaAPIWrapper,
    SQLDatabase,
)
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

mysql_xxl_job = SQLDatabase.from_uri(os.getenv("MYSQL_URI"))

llm = ChatOpenAI(model="deepseek-chat")


def search_web(query: str) -> str:
    """搜索网络"""
    params = {"engine": "bing", "gl": "us", "hl": "en", "q": query}
    search = SerpAPIWrapper(params=params)
    return search.run(query)


def search_wikipedia(query: str) -> str:
    """搜索维基百科"""
    search = WikipediaAPIWrapper()
    return search.run(query)
