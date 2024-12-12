import asyncio
import pandas as pd
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import nest_asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ProcessPoolExecutor
import sys

# Set recursion limit higher if needed
sys.setrecursionlimit(5000)

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("QDRANT_URL")

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)

# Check the status of the URL
def check_url_status(url):
    try:
        response = requests.head(url, allow_redirects=True)
        if response.status_code == 200:
            return True
        else:
            print(f"Skipping URL {url}: Received status code {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error while checking URL {url}: {e}")
        return False



# Async function to load a single URL
async def load_url(url):
    if check_url_status(url):  # Check if the URL is valid
        try:
            print("great ur here....")
            loader = WebBaseLoader(url, continue_on_failure=True)
            return await loader.aload()
        except RecursionError:
            print(f"RecursionError: Maximum recursion depth exceeded for URL: {url}")
        except Exception as e:
            print(f"Failed to load URL: {url}. Error: {e}")
    return None

# Async function to load all URLs

async def read_urls(urls):
    return await asyncio.gather(*(load_url(url) for url in urls))

# Function to read doc URLs from CSV
def read_docs(file_path):
    df = pd.read_csv(file_path)
    filenames = df['Doc_Urls'].values.tolist()
    return filenames

# Function to clean up the document content
def data_cleaning(docs):
    for doc in docs:
        if doc:  # Check if doc is not None
            doc.page_content = doc.page_content.replace('\n', '')
            doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
            doc.page_content = re.sub(r"[\"']", '', doc.page_content)
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
    return docs

# Function to create Qdrant vector store
def create_vector_qdrant(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)  # Reduced overlap
    texts = text_splitter.split_documents(docs)
    # Use OpenAI for embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

    qdrant = QdrantVectorStore.from_documents(
        texts,
        embeddings,
        url=url,
        api_key=api_key,
        collection_name="Docs_OpenAI",
        timeout=180
    )
    print("Dumped successfully...")
    return qdrant

# Function to chunk documents
def chunk_documents(doc_list, chunk_size):
    return [doc_list[i:i + chunk_size] for i in range(0, len(doc_list), chunk_size)]

# Async main function to orchestrate tasks
async def main():
    technical_docs = read_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Docs/technical_documentation.csv')
    docs = await read_urls(technical_docs)
    docs = data_cleaning(docs)

    chunk_size = 40  # Adjust chunk size as needed
    chunked_docs = chunk_documents(docs, chunk_size)
    print(f"Total chunks: {len(chunked_docs)}")

    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, chunk in enumerate(chunked_docs):
            print(f"Processing Chunk {idx + 1}:")
            futures.append(executor.submit(create_vector_qdrant, chunk))  # Submit task for parallel execution

        # Collect results
        for future in futures:
            future.result()  # Ensure all tasks are complete
        print("All tasks completed.")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
