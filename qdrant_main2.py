import asyncio
from typing import AsyncIterable
import os
import Qdrant_Database
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from search_directory import graph_retrieval
from pydantic import BaseModel
import uvicorn
import identify_source
import logging
import json
import re
import ast
import cache
from langchain_openai import OpenAIEmbeddings
import database
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import numpy as np
from langchain.schema import HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
import time
import functools

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

complete_response=[]
def measure_time(func):
    """Decorator to measure the execution time of a function."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    # Determine if the function is async or sync and apply the correct wrapper
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

@measure_time
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@measure_time
async def generate_context(store,query):
    compressor = FlashrankRerank(model='ms-marco-TinyBERT-L-2-v2', top_n=25)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=store.as_retriever(search_kwargs={"k": 15}, return_source_documents=True)
    )
    context = await asyncio.to_thread(compression_retriever.invoke, query)
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
    return context_text



class Message(BaseModel):
    content: str

@measure_time
async def send_message(uid, query, content: str) -> AsyncIterable[str]:
    complete_response = []  # List to hold the full response
    callback = AsyncIteratorCallbackHandler()
    chat_model = ChatOpenAI(
        model='gpt-4o',
        temperature=0.01,
        streaming=True,
        verbose=True,
        callbacks=[callback],
        timeout=30
    )

    # System message with instructions
    system_message_content = '''
    You are an expert in providing detailed, relevant responses and a citation expert, focused on ensuring accuracy and clarity in both content and citations.

    Response Structure:
        Your responses must be formatted with appropriate headings (no numbering).
        The insights should be detailed and elaborative, presenting relevant context and in-depth information.
        If there are no sources to cite please dont include it in your in text citations.
        Ensure that citations in the response text align strictly with the unique order of URLs in the "Sources" section.

    Citation Rules:
        If there are no sources in the context please dont cite sources.
        Integrate citations directly in the text, using sequential numbering starting from [1]("https://example.com") (e.g., [1]("https://example.com"), [2]("https://rediff.com")).
        Each citation must correspond to a unique source listed under the "Sources" section.
        Only sources cited in the text should be listed under "Sources."
        Ensure that the sources listed are unique and never repeated in the "Sources" section.
        In the "Sources" section, only URLs should be listed, with each URL presented in the order it appears in the text.

    Content Requirements for the "Sources" Section:
        Do not include sources if they are not present in the provided context
        Do not repeat or duplicate sources within the "Sources" section; each URL should appear only once in the list.
        Sources should be listed under the "Sources" heading, sequentially numbered starting from 1.
        If any source is mistakenly labeled as 'None,' remove it from the list.
        Each source should be listed only once, even if referenced multiple times in the response.

    Follow-Up Questions:
        After listing the sources, provide three follow-up questions under the heading "Follow Up Questions," focusing on further exploration of the topic.
    '''

    # Creating the asynchronous task for the model response generation
    task = asyncio.create_task(
        chat_model.agenerate(
            messages=[[SystemMessage(content=system_message_content), HumanMessage(content=content)]])
    )

    complete_text = ""  # Variable to accumulate the full response text

    try:
        async for token in callback.aiter():
            print(token)  # Real-time chunk logging
            complete_text += token  # Append to the complete response text
            yield token  # Yield the chunk as part of the real-time stream

    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()
         # Append the complete response to the list

    await task  # Wait for the task to complete



def parse_and_clean(data):
    # Extract the part after "Output: "
    list_string = data.split("Output: ")[1]

    # Convert the string representation of the list to an actual list
    parsed_list = ast.literal_eval(list_string)

    # Remove unnecessary quotes from each element
    cleaned_list = [item.replace('"', '').replace("'", "") for item in parsed_list]

    # Determine if it's single or multiple elements
    if len(cleaned_list) == 1:
        print("Single element:", cleaned_list[0])
    else:
        print("Multiple elements:", cleaned_list)

@measure_time
def identify_tool(query,chat_history):
    # query=re.sub(r"'", "", query)

    logging.info(f"Identifying tool for query: {query}")
    llm_output = identify_source.llm_response(query,chat_history)
    print(llm_output)
    print("Raw data content:", repr(llm_output))
    logging.info(f"LLM output: {llm_output}")
    data =  re.sub(r'^```json\n|```$', '', llm_output).strip()

    # Load the JSON data from the string
    try:
        # Parse the JSON string
        json_data = json.loads(data)

        # Extract needed information
        rephrased_query = json_data.get("Rephrased Query")
        output = json_data.get("Output")

        return rephrased_query, output

    except json.JSONDecodeError as e:
        # Log the JSON error and provide a message without stopping the app
        print("JSON decoding failed:", e)
        return None, None



@measure_time
def clean_empty(data):
    if isinstance(data, dict):
        return {
            k: clean_empty(v)
            for k, v in data.items()
            if v or v is False  # Keep non-empty values and boolean False values
        }
    elif isinstance(data, list):
        return [clean_empty(item) for item in data if item]
    else:
        return data


@measure_time
async def run_tool(rephrased_query, tool):
    """Identify the correct tool based on the query and run it asynchronously."""



    if rephrased_query and tool:
        logging.info(f"Rephrased Query: {rephrased_query}, Tool: {tool}")
        tasks = {}

        # Use a ThreadPoolExecutor for synchronous calls
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            directory_tasks = []
            global_resource_tasks = []

            # Collect tasks for Directory tool
            for key in tool:
                if key == "Directory":
                    print("ooolalala.....")
                    search_terms_dict = graph_retrieval_instance.extract_entities(rephrased_query)
                    search_terms = []

                    if "person" in search_terms_dict:
                        search_terms += search_terms_dict["person"]
                    if "organization" in search_terms_dict:
                        search_terms += search_terms_dict["organization"]

                    # Offload synchronous get_answer to the thread pool
                    task = loop.run_in_executor(executor, graph_retrieval_instance.get_answer, search_terms)
                    directory_tasks.append((key, task))

                # Collect tasks for global resources
                for dict_key in global_resources['store']:
                    if key=="Directory":
                        continue
                    if key in dict_key:
                        print("super-----------", key)
                        task = generate_context(global_resources['store'][dict_key], rephrased_query)
                        global_resource_tasks.append((key, task))

            # Run all tasks concurrently and collect results directly into `tasks`
            directory_results = await asyncio.gather(*(t[1] for t in directory_tasks))
            for (key, _), result in zip(directory_tasks, directory_results):
                tasks[key] = result

            global_resource_results = await asyncio.gather(*(t[1] for t in global_resource_tasks))
            for (key, _), result in zip(global_resource_tasks, global_resource_results):
                tasks[key] = result
        print(tasks)
        return tasks






@measure_time
async def stream_generator(tasks,query,uid):
    global complete_response
    buffer = ""
    cp=""

    # Check for "Directory" key in tasks and yield directory_data if found
    if "Directory" in tasks:
        print(tasks['Directory'])
        print(type(tasks['Directory']))
        directory_data = tasks["Directory"]
        directory_data = clean_empty(directory_data)
        print(type(directory_data))
        if directory_data:
            data_string = " ".join(f"{k} {v}" for k, v in directory_data.items())
            cp = data_string
            yield f"Directory Data: {directory_data}\n"


    for key, value in tasks.items():
        if key=="Directory":
            continue
        else:
            async for token in send_message(uid,query,value):
                buffer += token
                if token.startswith("**") and token.endswith(":") or token.endswith("?") or token.endswith(":") or token.endswith('    ') or token.endswith('\n'):
                    yield buffer.strip()
                    complete_response.append(buffer)
                    buffer = ""  # Reset buffer

    for key, value in tasks.items():
        if key == "Directory":
            continue
        elif  cp == "":
            cp=" ".join(complete_response)
        else:
            cp=cp+" ".join(complete_response)
            # cp="Response:"+cp
            # query="query:"+query
    await cache_instance.append_chat_history(uid,query,cp)
    complete_response=[]



class ChatMessage(BaseModel):
    uid: str
    question: str
    ai_msg: str

class RetrieveChatRequest(BaseModel):
    uid: str

@app.post("/append_chat/")
async def append_chat(chat: ChatMessage):
    """
    Append a chat message (question and AI response) to the user's chat history.
    """
    try:
        # Append chat to the cache
        response = await cache_instance.append_chat_history(chat.uid, chat.question, chat.ai_msg)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve_chat/")
async def retrieve_chat(data: RetrieveChatRequest):
    """
    Retrieve chat history for a specific user ID.
    """
    try:
        # Retrieve chat history from the cache
        history = await cache_instance.retrieve_chat_history(data.uid)
        if history:
            return {"status": "success", "chat_history": history}
        else:
            return {"status": "success", "chat_history": "No history found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/flush_chat/{uid}")
def flush_chat(uid: str):
    """
    Flush chat history for a specific user ID.
    """
    try:
        # Flush chat history from the cache
        response = cache_instance.flush_chat_history(uid)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class ProcessRequest(BaseModel):
    query: str
    uid: str
@measure_time
@app.post('/process')
async def process_query(request: ProcessRequest):
    query = request.query
    uid = request.uid
    print(query,uid)
    chat_history = await cache_instance.retrieve_chat_history(uid)
    if not chat_history:
        chat_history = []
    print(f'chat history for {uid} is {chat_history}')
    rephrased_query,tool = identify_tool(query,chat_history)
    print(rephrased_query,tool)
    tasks= await run_tool(rephrased_query, tool)
    if tasks:
        return StreamingResponse(stream_generator(tasks,query,uid), media_type="text/plain")
    else:
        return {"message": "No matching tool found or no response generated."}



if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)