from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig, RunnableWithMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import requests
import os

load_dotenv()

class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow API的嵌入实现"""
    
    def __init__(self, model="BAAI/bge-large-zh-v1.5", api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("SILICON_FLOW_API_KEY", "sk-jzbbqfdbnigbqofyuukrwlwaflogixndnjjvsdtljaxwedvt")
        self.url = "https://api.siliconflow.cn/v1/embeddings"
    
    def embed_documents(self, texts):
        """批量嵌入文档"""
        all_embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            all_embeddings.append(embedding)
        return all_embeddings
    
    def embed_query(self, text):
        """嵌入单个查询"""
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.request("POST", self.url, json=payload, headers=headers)
        response_data = response.json()
        
        # 根据API返回结构解析embedding
        if isinstance(response_data, dict) and "data" in response_data:
            return response_data["data"][0]["embedding"]
        else:
            raise ValueError(f"无法从响应中解析嵌入向量: {response_data}")

# 创建向量存储
def create_or_load_vectorstore():
    """创建或加载向量存储"""
    return Chroma(
        persist_directory="db",
        embedding_function=SiliconFlowEmbeddings(),
        collection_name="my_documents"
    )

# 添加示例文档
def add_sample_documents(vector_store):
    """向向量存储添加示例文档"""
    documents = [
        Document(page_content="LangChain是一个用于构建基于语言模型的应用程序的框架", metadata={"source": "doc1"}),
        Document(page_content="向量数据库用于存储和检索向量嵌入，以便进行语义搜索", metadata={"source": "doc2"}),
        Document(page_content="RAG(检索增强生成)通过检索相关文档来增强语言模型的回答", metadata={"source": "doc3"})
    ]
    vector_store.add_documents(documents)
    return vector_store

# 设置RAG聊天提示
def setup_rag_chain(vector_store):
    """设置RAG链"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    template = """你是一个有帮助的AI助手。
    
    使用以下检索到的上下文来回答用户的问题。如果你不知道答案，就说你不知道，不要编造信息。
    
    上下文:
    {context}
    
    历史对话:
    {chat_history}
    
    用户问题: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI(model="deepseek-chat")
    
    # 设置RAG链
    rag_chain = (
        {"context": retriever, "question": lambda x: x, "chat_history": lambda x: x.get("chat_history", [])}
        | prompt
        | model
    )
    
    return rag_chain

def get_session_history(user_id: str, conversation_id: str) -> RedisChatMessageHistory:
    """获取历史消息"""
    # 使用正确格式的Redis URL
    redis_url = os.getenv("REDIS_URL", "redis://:123456@localhost:6379/0")
    return RedisChatMessageHistory(session_id=f"{user_id}:{conversation_id}", url=redis_url)

def main():
    # 创建或加载向量存储
    vector_store = create_or_load_vectorstore()
    
    # 添加示例文档（如果需要）
    try:
        if vector_store.get()["ids"] == []:
            vector_store = add_sample_documents(vector_store)
    except Exception as e:
        print(f"检查向量库时出错: {e}")
        print("添加示例文档...")
        vector_store = add_sample_documents(vector_store)
    
    # 设置RAG链
    rag_chain = setup_rag_chain(vector_store)
    
    # 添加消息历史
    with_message_history = RunnableWithMessageHistory(
        rag_chain, 
        get_session_history, 
        input_messages_key='question',
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
                default="default_conversation_id",
                is_shared=True
            )
        ]
    )
    
    print("开始基于向量库的多轮对话！输入 'exit' 来结束对话。")
    
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'exit':
            break
        
        try:
            response = with_message_history.invoke(
                {"question": user_input},
                config=RunnableConfig(configurable={"user_id": "00001", "conversation_id": "00001"})
            )
            print("AI:", response.content)
        except Exception as e:
            print(f"处理请求时出错: {e}")

if __name__ == "__main__":
    main() 