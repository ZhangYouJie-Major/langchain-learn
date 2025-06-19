from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import os

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses the tools to answer the users' questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

model = ChatOpenAI(model="deepseek-chat")
chain = chat_prompt | model

def get_session_history(user_id: str, conversation_id: str) -> RedisChatMessageHistory:
    """
        获取历史消息
    :param user_id
    :param conversation_id
    :return:
    """
    return RedisChatMessageHistory(session_id=f"{user_id}:{conversation_id}", url=os.getenv("REDIS_URL"))


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

print("开始多轮对话！输入 'exit' 来结束对话。")

while True:
    user_input = input("你: ")
    if user_input.lower() == 'exit':
        break

    response = with_message_history.invoke(
        {"question": f"{user_input}"},
        config=RunnableConfig(configurable={"user_id": "00001", "conversation_id": "00001"})
    )
    print(response)