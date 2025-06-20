import os
import asyncio
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from pydantic import SecretStr

openai_api_key = SecretStr("sk-6cc44ac6764046bbb5520195e40b14aa")
open_ai_url = "https://api.deepseek.com"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=openai_api_key,
    base_url=open_ai_url,
)

prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}笑话")

parser = StrOutputParser()

chain = prompt | llm


async def async_stream1():
    chunks = []
    async for chunk in chain.astream({"topic": '冷'}):
        chunks.append(chunk)
        if len(chunks) == 2:
            print(chunks[1])
        print(chunk.content, end="|", flush=True)


async def async_stream2():
    chunks = []

    async for chunk in chain.astream({"topic": '古典'}):
        chunks.append(chunk)
        if len(chunks) == 2:
            print(chunks[1])
        print(chunk.content, end="|", flush=True)


responses = chain.batch([{"topic": '冷'}, {"topic": '古典'}])
for response in responses:
    print(response.content)



