import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.embeddings.siliconflow import SiliconFlowEmbeddings
from dotenv import load_dotenv

def load_pdf_to_documents(pdf_path):
    """加载 PDF 文件为 Document 列表"""
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    return docs

def create_or_load_vectorstore():
    """创建或加载向量存储"""
    return Chroma(
        persist_directory="db",
        embedding_function=SiliconFlowEmbeddings(),
        collection_name="my_documents"
    )

def add_pdf_to_vectorstore(pdf_path, vector_store):
    """将 PDF 文档嵌入并写入向量库"""
    docs = load_pdf_to_documents(pdf_path)
    if docs:
        vector_store.add_documents(docs)
        print(f"已添加 {len(docs)} 段 PDF 文档到向量库。")
    else:
        print("未能加载到任何文档内容。")

def main():
    load_dotenv()
    pdf_path = "/Users/zhangyoujie/Desktop/学习/500道Java后端面试必知必会-V1版.pdf"  # 修改为你的 PDF 路径
    vector_store = create_or_load_vectorstore()
    add_pdf_to_vectorstore(pdf_path, vector_store)

if __name__ == "__main__":
    main() 