from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

from dotenv import load_dotenv

from pydantic import Field, BaseModel

api_wrapper = WikipediaAPIWrapper()


class CustomWikiInput(BaseModel):
    query: str = Field(description="The query to search the web for")


tool = WikipediaQueryRun(
    name="wiki-tool",
    description="A tool for searching the web for information",
    args_schema=CustomWikiInput,
    api_wrapper=api_wrapper,
    return_direct=True,
)

print(tool.invoke("奥本海默影片"))
