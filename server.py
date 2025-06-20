from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.globals import set_debug,set_verbose
from app.schema.agent_schema import AgentInput, AgentOutput
from tool.google_search_tool import google_search
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
           Thought:{agent_scratchpad}
           '''

chat_template = '''
你是一个问答小助手,帮我回答问题 {question}
'''

agent_prompt = PromptTemplate.from_template(template)
chat_prompt = PromptTemplate.from_template(chat_template)

app = FastAPI()

model = ChatOpenAI(model="deepseek-chat")
tools = [google_search]

agent = create_react_agent(model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
set_debug(True)

# 创建链：提示模板 + LLM
chain = chat_prompt | model


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, chain, path='/chain', input_type=AgentInput,
           output_type=AgentOutput)
add_routes(app, agent_executor, path='/agent', input_type=AgentInput,
           output_type=AgentOutput)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
