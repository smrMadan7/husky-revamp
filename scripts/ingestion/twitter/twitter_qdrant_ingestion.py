import argparse
import json
import os
import re

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("QDRANT_URL")
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)


def read_files(files):
    """
    Reads JSON files and extracts Twitter data into Document objects.

    Args:
        files (list): List of file paths to be read.

    Returns:
        list: A list of Document objects containing the text and metadata from the files.
    """
    data = " "
    docs = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        for item in data["twitter_data"]:
            text = {value for key, value in item.items() if key == "text"}
            metadata = {key: value for key, value in item.items() if key != "text"}
            doc = Document(page_content=str(text), metadata=metadata)
            docs.append(doc)
    return docs


def create_vector_qdrant(docs, embeddings_model, collection_name):
    """
    Creates a vector store in Qdrant from document chunks using specified embedding models.

    Args:
        docs (list): List of Document objects to be embedded and stored.
        embeddings_model (str): The type of embedding model to use ('OpenAI', 'OpenSource', etc.).
        collection_name (str): Name of the Qdrant collection.

    Returns:
        QdrantVectorStore: The created vector store object.
    """
    retrieval_mode = ""
    sparse_embeddings = None  # Default to None to handle optional cases

    if embeddings_model == "OpenAI":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        retrieval_mode = RetrievalMode.DENSE

    elif embeddings_model == "OpenSource":
        embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en", model_kwargs={"device": "cpu"})
        retrieval_mode = RetrievalMode.DENSE

    elif embeddings_model == "OpenAI-Hybrid":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
        retrieval_mode = RetrievalMode.HYBRID

    elif embeddings_model == "OpenSource-Hybrid":
        embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en", model_kwargs={"device": "cpu"})
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
        retrieval_mode = RetrievalMode.HYBRID

    elif embeddings_model == "Sparse":
        embeddings = FastEmbedSparse(model_name="Qdrant/BM25")
        retrieval_mode = RetrievalMode.SPARSE

    else:
        raise ValueError("Invalid embeddings_model provided.")

    # Conditional argument passing for sparse embeddings
    qdrant_args = {"documents": docs, "embedding": embeddings, "url": url, "api_key": api_key, "collection_name": collection_name, "retrieval_mode": retrieval_mode, "timeout": 360}

    if sparse_embeddings:
        qdrant_args["sparse_embedding"] = sparse_embeddings

    qdrant = QdrantVectorStore.from_documents(**qdrant_args)
    print("Dumped successfully...")
    return qdrant


def read_file_path(folder_path):
    """
    Reads all JSON file paths in a specified folder.

    Args:
        folder_path (str): Path to the folder containing the JSON files.

    Returns:
        list: A list of file paths.
    """
    files = os.listdir(folder_path)
    files = [os.path.join(folder_path, file) for file in files if ".json" in file]
    return files


def data_cleaning(docs):
    """
    Cleans the content of documents by removing newlines and special characters.

    Args:
        docs (list): List of Document objects to be cleaned.

    Returns:
        list: List of cleaned Document objects.
    """
    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", "")
        doc.page_content = re.sub(r'[",\\]', "", doc.page_content)
        doc.page_content = re.sub(r"[\"']", "", doc.page_content)
        doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
    return docs


def chunk_documents(doc_list, chunk_size):
    """
    Splits a list of documents into smaller chunks.

    Args:
        doc_list (list): List of Document objects to be chunked.
        chunk_size (int): Number of documents per chunk.

    Returns:
        list: List of chunks, where each chunk is a list of Document objects.
    """
    return [doc_list[i : i + chunk_size] for i in range(0, len(doc_list), chunk_size)]


def main(folder_path, collection_name, embedding_model, chunk_size):
    """
    Main function to read, clean, and process documents and create vectors for Qdrant.

    Args:
        folder_path (str): Path to the folder containing the documents.
        collection_name (str): Name of the Qdrant collection.
        embedding_model (str): The type of embedding model to use ('OpenAI', 'OpenSource', etc.).
        chunk_size (int): Size of each chunk for processing.
    """
    filenames = read_file_path(folder_path)
    docs = read_files(filenames)
    docs = data_cleaning(docs)
    chunked_docs = chunk_documents(docs, chunk_size)

    print(f"Total number of chunks: {len(chunked_docs)}")
    for idx, chunk in enumerate(chunked_docs):
        print(f"Chunk {idx + 1}:")
        create_vector_qdrant(chunk, embedding_model, collection_name)
        print("*" * 180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and create vectors for Qdrant.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing documents.")
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the Qdrant collection.")
    parser.add_argument("--embedding_model", type=str, required=True, choices=["OpenAI", "OpenSource", "OpenAI-Hybrid", "OpenSource-Hybrid", "Sparse"], help="Type of embedding model to use.")
    parser.add_argument("--chunk_size", type=int, default=40, help="Size of each chunk for document processing.")

    args = parser.parse_args()

    # Call main with the parsed arguments
    main(args.folder_path, args.collection_name, args.embedding_model, args.chunk_size)


# sample usage

# python twitter_qdrant_ingestion.py --folder_path '/home/ubuntu/Downloads/husky-be-v1/Data/Teams_Twitter' --collection_name 'Twitter_OpenAI_Trial' --embedding_model 'OpenAI' --chunk_size 40
