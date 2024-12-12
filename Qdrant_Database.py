import os
import time
import asyncio
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import RetrievalMode
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

class QdrantStorageTest:
    def __init__(self, collection_names):
        self.collection_names = collection_names
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=self.openai_api_key)
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.stores = {}
        self.keep_alive_interval = 300  # Default to 1 hour

    def connect_to_collection(self, collection_name):
        try:
            logging.info(f"Establishing Qdrant connection for {collection_name}")
            store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=collection_name,
                retrieval_mode=RetrievalMode.DENSE,
                timeout=self.keep_alive_interval
            )
            self.stores[collection_name] = store
            logging.info(f"QdrantVectorStore initialized for {collection_name}")
        except Exception as e:
            logging.error(f"Error initializing QdrantVectorStore for {collection_name}: {e}")

    def connect_storage(self):
        start_time = time.time()
        max_workers = min(32, len(self.collection_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.connect_to_collection, name): name for name in self.collection_names}
            for future in as_completed(futures):
                collection_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error connecting to {collection_name}: {e}")

        end_time = time.time()
        logging.info(f"Total time to connect storage: {end_time - start_time:.2f} seconds")
        return self.stores

    def keep_alive_collection(self, collection_name):
        if self.client:
            return self.client.get_collection(collection_name)
        else:
            raise Exception("Client is not initialized.")

    async def keep_alive(self):
        while True:
            for collection_name in self.collection_names:
                retry_attempts = 3
                for attempt in range(retry_attempts):
                    try:
                        print(" i am checking the qdrant db connection...")
                        collection_info = self.keep_alive_collection(collection_name)
                        logging.info(f"Keep-alive query result for {collection_name}: {collection_info}")
                        if collection_info.status=='green':
                            break  # Break on successful keep-alive
                    except Exception as e:
                        logging.error(f"Keep-alive attempt {attempt + 1} failed for {collection_name}: {e}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2)  # Wait before retrying
                        else:
                            logging.error(f"All keep-alive attempts failed for {collection_name}.")
            await asyncio.sleep(self.keep_alive_interval)

    def start_keep_alive(self):
        asyncio.create_task(self.keep_alive())
