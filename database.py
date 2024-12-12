import os
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import time


class Storage:
    url = None
    username = None
    password = None
    model = None
    _store = None  # Global variable to hold the connection

    def __init__(self):
        try:
            self.url = os.environ['neo4j_url']
            self.username = os.environ['neo4j_username']
            self.password = os.environ['neo4j_password']
            self.model = os.environ['model']
            self.index_name = os.environ['index_name']
            self.text_node_property = os.environ['text_node_property']
            self.embedding_node_property = os.environ['embedding_node_property']
            self.node_label = os.environ['node_label']
        except Exception as e:
            print(e)
            raise e

    def connect_storage(self):
        if Storage._store is not None:
            print("Reusing existing DB connection", time.time() * 1000)
            return Storage._store
        try:
            print("ESTABLISHING DB CONNECTION", time.time() * 1000)
            Storage._store = Neo4jVector.from_existing_index(
                OpenAIEmbeddings(model=self.model),
                url=self.url,
                username=self.username,
                password=self.password,
                index_name=self.index_name,
                text_node_property=self.text_node_property,
                embedding_node_property=self.embedding_node_property,
                node_label=self.node_label,
            )
            print("DB CONNECTION ESTABLISHED", time.time() * 1000)
        except Exception as e:
            print(f"Error establishing connection: {e}")
            return {"error": "Error establishing connection with Neo4j"}

        return Storage._store

