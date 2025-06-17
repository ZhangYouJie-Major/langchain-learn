import os

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

openai_api_key = SecretStr("sk-6cc44ac6764046bbb5520195e40b14aa")
open_ai_url = "https://api.deepseek.com"

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个世界级的技术专家"),
    ("human", "{input}"),
])
str_template = PromptTemplate.from_template("给我讲一个关于{content}的{type}笑话")
prompt = str_template.format_prompt(content="机器学习", type="笑话")

message = chat_template.format_messages(input = "请写一个关于机器学习的文章")
print(prompt)

out_parser = StrOutputParser()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=openai_api_key,
    base_url=open_ai_url,
)

chain = str_template | llm | out_parser

res = chain.invoke({"input": "帮我检写一篇关于ai的技术文章，100字"})
print(res)
