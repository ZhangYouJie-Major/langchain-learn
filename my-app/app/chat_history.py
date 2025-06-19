from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.globals import set_debug, set_verbose
from langchain_core.runnables import ConfigurableFieldSpec
from agent_schema import AgentInput, AgentOutput
from tool.google_search_tool import google_search
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses the tools to answer the users' questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

model = ChatOpenAI(model="deepseek-chat")
chain = chat_prompt | model

store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    """
        获取历史消息
    :param user_id
    :param conversation_id
    :return:
    """
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key='question',
                                                  history_messages_key='chat_history',
                                                  history_factory_config=[
                                                      ConfigurableFieldSpec(
                                                          id='user_id',
                                                          annotation=str,
                                                          name="user_id",
                                                          description="用户id",
                                                          default="default_user_id",
                                                          is_shared=True
                                                      ),
                                                      ConfigurableFieldSpec(
                                                          id='conversation_id',
                                                          annotation=str,
                                                          name="conversation_id",
                                                          description="会话id",
                                                          default="会话id",
                                                          is_shared=True
                                                      )
                                                  ])
# 调用with_message_history
response = with_message_history.invoke(
    {"question": "什么是余弦函数"},
    config=RunnableConfig(configurable={"user_id": "abc2", "conversation_id": "abc2"})
)
print(response)

# 调用with_message_history
response1 = with_message_history.invoke(
    {"question": "没听懂"},
    config=RunnableConfig(configurable={"user_id": "abc2", "conversation_id": "abc2"})
)
print(response1)

# 调用with_message_history
response2 = with_message_history.invoke(
    {"question": "没听懂"},
    config=RunnableConfig(configurable={"user_id": "abc2", "conversation_id": "11111"})
)
print(response2)
