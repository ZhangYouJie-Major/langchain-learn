from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_core.output_parsers import StrOutputParser
from agent_schema import AgentInput, AgentOutput
from dotenv import load_dotenv

load_dotenv()

template = '''Answer the following questions as best you can. You have access to the following tools:

           {tools}

           Use the following format:

           Question: the input question you must answer
           Thought: you should always think about what to do
           Action: the action to take, should be one of [{tool_names}]
           Action Input: the input to the action
           Observation: the result of the action
           ... (this Thought/Action/Action Input/Observation can repeat N times)
           Thought: I now know the final answer
           Final Answer: the final answer to the original input question

           Begin!

           Question: {question}
           Thought:{agent_scratchpad}'''

chat_template = '''
将以下句子翻译成英文：{sentence}
'''

agent_prompt = PromptTemplate.from_template(template)
chat_prompt = PromptTemplate.from_template(chat_template)


@tool
def search(query: str) -> str:
    """
     从google查询答案
    :param query:  问题
    :return:  回答
    """
    print('mcp工具被调用')
    return f"搜索结果：{query}"


app = FastAPI()

model = ChatOpenAI(model="deepseek-chat")
tools = []
agent_parse = JSONAgentOutputParser()

agent = create_react_agent(model, tools, agent_prompt, output_parser=agent_parse)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 创建链：提示模板 + LLM
chain = chat_prompt | model


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, agent_executor, path='/agent', input_type=AgentInput,
           output_type=AgentOutput)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
