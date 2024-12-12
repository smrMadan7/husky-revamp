from fastapi import FastAPI, HTTPException
import asyncio
import logging
import json
import re
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import Qdrant_Database
import llm_qdrant
from pydantic import BaseModel
import database
from search_directory import graph_retrieval
import cache
import identify_source
import uvicorn
import asyncio
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from openai import OpenAI
import numpy as np
from fastapi.responses import StreamingResponse
import time
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Loading environment variables
collection_names_str = os.getenv("Collection_Names", "").strip()
collection_names = collection_names_str.split(",")
cache_instance = cache.cache()
storage_instance = database.Storage()
graph_retrieval_instance = graph_retrieval()

# Dictionary to store connections and embeddings
global_resources = {
    "storage": None,
    "store": None,
    "embeddings": None,
    "neo4j_storage": None,
    "neo4j_store": None
}

def initialize_qdrant():
    """Function to initialize the Qdrant database connection and embeddings."""
    global_resources["storage"] = Qdrant_Database.QdrantStorageTest(collection_names)
    global_resources["store"] = global_resources["storage"].connect_storage()
    global_resources["embeddings"] = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"))
    global_resources['neo4j_store'] = storage_instance.connect_storage()
    logging.info("Database connection and embeddings initialized at startup")

app = FastAPI(on_startup=[initialize_qdrant])


def identify_tool(query):
    logging.info(f"Identifying tool for query: {query}")
    llm_output = identify_source.llm_response(query)
    logging.info(f"LLM output: {llm_output}")

    match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
    if match:
        json_string = match.group(0)
        try:
            json_string = json_string.replace("'", '"')
            json_data = json.loads(json_string)
            rephrased_query = json_data.get('Rephrased Query', '')
            tool = json_data.get('Output', [])
            logging.info(f"Identified tool: {tool}")
            return rephrased_query, tool
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error: {e}")
    else:
        logging.warning("No JSON object found in response.")
    return None, None
def convert_floats(obj):
    """Recursively convert np.float32 to float."""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    return obj


async def llm_response(query, context):
    client =  OpenAI()
    context_serialized = [
        {
            'page_content': doc.page_content,
            'metadata': {
                'source': doc.metadata.pop('url', None),
                **{k: float(v) if isinstance(v, np.float32) else v for k, v in doc.metadata.items()}
            }
        }
        for doc in context
    ]
    context_text = query + "\n\n".join(
        [f"Source: {doc['metadata'].get('source', 'N/A')}\n\nContent: {doc['page_content']}" for doc in
         context_serialized]
    )

    try:
        # Use a standard completion call instead of streaming
        collected_chunks = []
        collected_messages = []
        response = await asyncio.to_thread(client.chat.completions.create,
                                           model='gpt-4o',
                                           messages=[{
                                               "role": "system",
                                               "content": (
                                                   "You are an expert in providing detailed and relevant responses based on the query and given context and a citation expert. "
                                                   "Properly format your detailed responses under appropriate headings using citations inside text wherever appropriate without fail. "
                                                   "Please provide all unique sources based on which you generated the response. "
                                                   "Sources should appear after the response is complete under the heading Sources. They should be numbered, unique and should contain only the urls not the titles "
                                                   "and should not repeat. If any source ends with 'None', remove it."
                                                   "Based on your response please provide 3 relevant questions under the heading Follow Up Questions"
                                                   "Provide the response as shown in the Example"
                                                   "Example:"
                                                   " ### Filecoin News 93 - **Filecoin and Storacha Collaboration**:"
                                                    "Filecoin has partnered with Storacha to enhance decentralized storage solutions. This collaboration integrates Storacha’s high-performance storage architecture with Filecoin’s decentralized network, offering faster and more reliable data storage solutions for developers, businesses, and creators [1](https://filecoin.io/blog/posts/filecoin-news-93/)."

                                               )
                                           }, {
                                               "role": "user",
                                               "content": context_text,
                                           }],
                                           temperature=0,
                                           max_tokens=4096,
                                           stream=True)

        async def event_stream():
            start_time = time.time()
            chunk_time=0
            for chunk in response:
                chunk_time = time.time() - start_time  # calculate the time delay of the chunk
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk.choices[0].delta.content  # extract the message
                if chunk_message is not None:
                    collected_messages.append(chunk_message)  # save the message
                    yield f"{chunk_message}\n"  # Yield the chunk message
                    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")

            # After the loop, combine the collected messages if needed
            full_reply_content = ''.join(collected_messages)
            print(f"Full response received {chunk_time:.2f} seconds after request")
            print("Full message content:", full_reply_content)

        # Return the StreamingResponse, streaming each chunk as it arrives
        return StreamingResponse(event_stream(), media_type="text/plain")
    except asyncio.TimeoutError:
        return {"error": "Timeout", "message": "Request took too long to complete."}

    except Exception as e:
        return {"error": "Execution Error", "message": str(e)}


async def respond(store,query):
    compressor = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2', top_n=15)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=store.as_retriever(search_kwargs={"k": 10}, return_source_documents=True)
    )
    context = await asyncio.to_thread(compression_retriever.invoke, query)
    return await llm_response(query, context)



async def identify_and_run_tool(query: str):
    """Identify the correct tool based on the query and run it asynchronously."""
    results = {}
    directory_data = {}

    rephrased_query, tool = identify_tool(query)

    if rephrased_query and tool:
        logging.info(f"Rephrased Query: {rephrased_query}, Tool: {tool}")
        tasks = {}

        # Collect tasks based on identified tools
        for key in tool:
            if key == "Directory":
                search_terms_dict = graph_retrieval_instance.extract_entities(rephrased_query)
                search_terms = []

                if "person" in search_terms_dict:
                    search_terms += search_terms_dict["person"]
                if "organization" in search_terms_dict:
                    search_terms += search_terms_dict["organization"]

                directory_data = graph_retrieval_instance.get_answer(search_terms)

            # Check if the store exists in global_resources
            for dict_key in global_resources['store']:
                if key in dict_key:
                    print("super-----------",key)
                    tasks[key] = await respond(global_resources['store'][dict_key], rephrased_query)

        # Await all tasks and gather results
        if tasks:
            results = await asyncio.gather(*tasks.values())
            results = dict(zip(tasks.keys(), results))  # Use task keys to store results in a dictionary


        return results, directory_data

    else:
        # If tool identification fails or rephrased query is empty
        logging.error("Tool identification failed.")
        return {"error": "Tool identification failed"}, {}

class ProcessRequest(BaseModel):
    query: str
    uid: str

@app.post('/process')
async def process_query(request: ProcessRequest):
    query = request.query
    uid = request.uid
    print(query)
    logging.info(f"Received query: {query}, UID: {uid}")

    # Retrieve chat history from cache (if relevant)
    chat_history = await cache_instance.retrieve_chat_history(uid) or []

    try:
        # Identify and run the tool for the query
        results, directory_data = await identify_and_run_tool(query)

        # Return LLM output directly without streaming
        return {
            "results": results,
            "directory_data": directory_data
        }

    except Exception as e:
        logging.error(f"Error while processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Execution Error")

if __name__ == "__main__":
    import subprocess

    subprocess.run([
        "uvicorn",
        "qdrant_main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ])
    # uvicorn.run(qdrant_main, host="127.0.0.1", port=8000, log_level="info",reload=True)
