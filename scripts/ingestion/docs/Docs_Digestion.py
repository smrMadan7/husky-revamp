import asyncio
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader,Docx2txtLoader,UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import nest_asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("QDRANT_URL")
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)




def read_urls(urls):
    nest_asyncio.apply()
    loader = WebBaseLoader(urls,continue_on_failure=True)
    loader.requests_per_second = 1
    data = loader.aload()  # Await the coroutine
    return data


def read_docs(file_path):
    df=pd.read_csv(file_path)
    filenames=df['Doc_Urls'].values.tolist()
    return filenames

def data_cleaning(docs):
    for doc in docs:
        doc.page_content = doc.page_content.replace('\n', '')
        doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
        doc.page_content = re.sub(r"[\"']", '', doc.page_content)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
    return docs


def create_vector_qdrant(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en", model_kwargs={'device': 'cpu'})
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")

    qdrant = QdrantVectorStore.from_documents(
        texts,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=url,
        api_key=api_key,
        collection_name="Docs",
        retrieval_mode=RetrievalMode.HYBRID,
        timeout=360
    )
    print("Dumped successfully...")
    return qdrant

def chunk_documents(doc_list, chunk_size):
    return [doc_list[i:i + chunk_size] for i in range(0, len(doc_list), chunk_size)]





if __name__=="__main__":
    technical_docs=read_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Docs/technical_documentation1.csv')
    docs=read_urls(technical_docs)
    docs=data_cleaning(docs)
    chunk_size = 40
    chunked_docs = chunk_documents(docs, chunk_size)
    print(len(chunked_docs))
    for idx, chunk in enumerate(chunked_docs):
        print(f"Chunk {idx + 1}:")
        create_vector_qdrant(chunk)
        print("*"*180)

