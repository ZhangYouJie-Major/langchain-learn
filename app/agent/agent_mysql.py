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

toolkit = SQLDatabaseToolkit(db=mysql_xxl_job, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="zero-shot-react-description",
    verbose=True,
)

result = agent_executor.invoke({"input": "查询xxl-job的用户信息"})
print(result)
