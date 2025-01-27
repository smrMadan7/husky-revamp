import asyncio
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
from groq import Groq
import nest_asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode
import argparse
from typing import List

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("QDRANT_URL")

qdrant_client = QdrantClient(url=url, api_key=api_key)


def read_urls(urls: List[str]):
    try:
        nest_asyncio.apply()
        loader = WebBaseLoader(urls, continue_on_failure=True)
        loader.requests_per_second = 1
        data = loader.aload()
        return data
    except Exception as e:
        print(f"Error reading URLs: {e}")
        return []


def read_docs(file_path: str):
    try:
        df = pd.read_csv(file_path)
        df = df[~(df['Exists'] == True)]
        filenames = df['Doc_Urls'].values.tolist()
        return filenames
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading document file: {e}")
        return []


def extract_date(context: str):
    try:
        client = Groq()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify and provide only the publication date mentioned in the context and do not mention any event date. "
                        "If there are multiple dates provide only the first occurring date. "
                        "Respond with the date alone in 'Month Day' or 'Month Day' format without any additional text. "
                        "If Month is in abbreviated form, please expand and provide the response. "
                        "Example: Oct 5 should be October 5."
                    )
                },
                {
                    "role": "user",
                    "content": context,
                }
            ],
            model="gemma2-9b-it",
            temperature=0.01,
            max_tokens=7000,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error extracting date: {e}")
        return ""


def normalize_dates_in_string(date_string: str):
    try:
        updated_string = re.sub(
            r'(\b[A-Za-z]+\s\d{1,2}\b)(?!,\s\d{4})',
            r'\1 2024',
            date_string
        )
        return updated_string
    except Exception as e:
        print(f"Error normalizing dates: {e}")
        return date_string


def data_cleaning(docs):
    try:
        for doc in docs:
            try:
                date = extract_date(doc.page_content)
                date = normalize_dates_in_string(date)
                doc.metadata['publication_date'] = date
                doc.page_content = re.sub(r'[\n"]', '', doc.page_content)
                doc.page_content = re.sub(r"[\"']", '', doc.page_content)
                doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
            except Exception as e:
                print(f"Error cleaning document: {doc.metadata.get('source', 'unknown source')}, {e}")
        return docs
    except Exception as e:
        print(f"Error in data cleaning: {e}")
        return docs


def create_vector_qdrant(docs, embeddings_model: str, collection_name: str):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        retrieval_mode = ""
        sparse_embeddings = None

        if embeddings_model == "OpenAI":
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
            retrieval_mode = RetrievalMode.DENSE

        elif embeddings_model == "OpenSource":
            embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en",
                                               model_kwargs={"device": "cpu"})
            retrieval_mode = RetrievalMode.DENSE

        elif embeddings_model == "OpenAI-Hybrid":
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
            retrieval_mode = RetrievalMode.HYBRID

        elif embeddings_model == "OpenSource-Hybrid":
            embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en",
                                               model_kwargs={"device": "cpu"})
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
            retrieval_mode = RetrievalMode.HYBRID

        elif embeddings_model == "Sparse":
            embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
            retrieval_mode = RetrievalMode.SPARSE

        else:
            raise ValueError("Invalid embeddings_model provided.")

        qdrant_args = {
            "documents": texts,
            "embedding": embeddings,
            "url": url,
            "api_key": api_key,
            "collection_name": collection_name,
            "retrieval_mode": retrieval_mode,
            "timeout": 360
        }

        if sparse_embeddings:
            qdrant_args["sparse_embedding"] = sparse_embeddings

        qdrant = QdrantVectorStore.from_documents(**qdrant_args)
        print("Dumped successfully...")
        return qdrant

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


def chunk_documents(doc_list, chunk_size):
    return [doc_list[i:i + chunk_size] for i in range(0, len(doc_list), chunk_size)]


def main(file_path: str, collection_name: str, embedding_model: str, chunk_size: int):
    try:
        filenames = read_docs(file_path)
        if not filenames:
            print("No documents to process.")
            return

        docs = read_urls(filenames)
        if not docs:
            print("No data fetched from URLs.")
            return

        docs = data_cleaning(docs)
        chunked_docs = chunk_documents(docs, chunk_size)
        print(f"Total number of chunks: {len(chunked_docs)}")

        for idx, chunk in enumerate(chunked_docs):
            print(f"Processing chunk {idx + 1}...")
            create_vector_qdrant(chunk, embedding_model, collection_name)
            print("*" * 180)

    except Exception as e:
        print(f"Error in main function: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and create vectors for Qdrant.")
    parser.add_argument("--documents_file_path", type=str, required=True,
                        help="Path to the folder containing documents.")
    parser.add_argument("--collection_name", type=str, required=True,
                        help="Name of the Qdrant collection.")
    parser.add_argument("--embedding_model", type=str, required=True,
                        choices=["OpenAI", "OpenSource", "OpenAI-Hybrid", "OpenSource-Hybrid", "Sparse"],
                        help="Type of embedding model to use.")
    parser.add_argument("--chunk_size", type=int, default=40,
                        help="Size of each chunk for document processing.")
    args = parser.parse_args()

    main(args.documents_file_path, args.collection_name, args.embedding_model, args.chunk_size)
