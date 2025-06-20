import os
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from transformers import AutoTokenizer

class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow API的嵌入实现"""
    
    def __init__(self, model="BAAI/bge-large-zh-v1.5"):
        self.model = model
        self.api_key = os.getenv('SILICON_FLOW_API_KEY')
        self.url = os.getenv('SILICON_FLOW_API_URL')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        except Exception as e:
            raise RuntimeError(f"分词器加载失败: {e}. 请确保 transformers 已安装且模型名称正确。");
    
    def embed_documents(self, texts):
        """批量嵌入文档"""
        all_embeddings = []
        for text in texts:
            # 处理 Document 或其他类型
            if hasattr(text, "page_content"):
                text = text.page_content
            elif not isinstance(text, str):
                text = str(text)
            embedding = self._get_embedding(text)
            all_embeddings.append(embedding)
        return all_embeddings
    
    def embed_query(self, text):
        """嵌入单个查询"""
        # 处理各种类型
        if isinstance(text, BaseMessage):
            text = text.content
        elif isinstance(text, list) and len(text) > 0:
            first = text[0]
            if isinstance(first, BaseMessage):
                text = first.content
            else:
                text = str(first)
        elif not isinstance(text, str):
            text = str(text)
        return self._get_embedding(text)
    
    def _truncate_text(self, text, max_tokens=128):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        while len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return text
    
    def _get_embedding(self, text):
        if hasattr(text, "page_content"):
            text = text.page_content
        elif not isinstance(text, str):
            text = str(text)
        text = self._truncate_text(text)
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
        try:
            response_data = response.json()
        except Exception as e:
            raise ValueError(f"API响应不是合法JSON: {e}, 响应内容: {response.text}")
        
        # 根据API返回结构解析embedding
        if isinstance(response_data, dict) and "data" in response_data and response_data["data"]:
            print(response_data)
            return response_data["data"][0]["embedding"]
        elif isinstance(response_data, dict) and "message" in response_data:
            raise ValueError(f"API返回错误: {response_data['message']}")
        else:
            raise ValueError(f"无法从响应中解析嵌入向量: {response_data}") 