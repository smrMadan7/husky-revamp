from fastapi import FastAPI, HTTPException,Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import asyncio
import uvicorn
from dotenv import load_dotenv
from models.feedback import FeedbackModel
from schema.feedback import FeedbackSchema
from services.feedback import add_feedback
import parallel_chain3
import twitter_bot
import traceback
from cache import cache
import re
from typing import Optional
import httpx
from retrieve_graph_v2 import graph_retrieval
from fastapi import FastAPI, HTTPException,Request, Depends
from schema.conversation import ConversationSchema
from services.conversation import add_conversation
import twitter_search
from database import Storage
import os
# Load environment variables from .env file
load_dotenv()

app = FastAPI(debug=True)

# CORS Middleware must be added first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can restrict to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

         # List of paths to exclude from authentication
        public_paths = ["/"]
        
        if request.method == "OPTIONS":
            return await call_next(request)

        # Skip middleware for public paths
        if request.url.path in public_paths:
            return await call_next(request)

        origin = request.headers.get("Origin")
        if origin == "https://husky-poc-web.vercel.app":
            return await call_next(request)

        # Extract token from Authorization header
        auth_header = request.headers.get('Authorization')
        print(f"Authorization Header: {auth_header}")  # Debugging print statement
        
        if auth_header is None:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: Missing Authorization header"})
        
        # Parse Bearer token
        scheme, _, token = auth_header.partition(' ')
        if scheme.lower() != 'bearer' or not token:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid token format"})
        
        # Verify JWT Token with external API
        try:

            if origin == os.getenv("STAGE_DOMAIN"):
                auth_api_url = os.getenv("STAGE_AUTH_URL")
            else:
                auth_api_url = os.getenv("AUTH_URL")
            print(origin,auth_api_url)
            if not auth_api_url:
                raise ValueError("AUTH_URL environment variable is not set.")

            # Send a POST request to the external API to verify the token
            async with httpx.AsyncClient() as client:
                response = await client.post(auth_api_url, json={"token": token})
                response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx

                data = response.json()  # Parse the JSON response
                print(data)
                if data.get("active"):  # Check if the token is active based on your API response
                    response = await call_next(request)  # Proceed to the next middleware or request handler
                    return response
                else:
                    return JSONResponse(status_code=401, content={"detail": "Unauthorized: Token inactive or invalid"})

        except httpx.HTTPStatusError as http_err:
            print(f"HTTP error occurred: {http_err}")  # Log the error for debugging
            return JSONResponse(status_code=http_err.response.status_code, content={"detail": str(http_err)})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")  # Log unexpected errors
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# app.add_middleware(AuthMiddleware)


storage_instance = Storage()
# Initialize the cache instance
cache_instance = cache()

async def get_answer_from_twitter_bot(query, chat_history):
    await asyncio.sleep(2)
    return "Twitter answer", ["source2"], {"questions": ["Follow-up question 2"]}, ["augmented data"]

@app.post('/process')
async def process_query(uid, query, source='None'):
    store = storage_instance.connect_storage()
    if "error" in store:
        raise HTTPException(status_code=500, detail=store["error"])
    response = {}
    chat_history = await cache_instance.retrieve_chat_history(uid)  # Use cache_instance

    if not chat_history:
        chat_history = []

    print(uid, query, source)

    answer = sources = followup_qns = augmentation = None

    if ((source == "None") or (source == "none")):
        chatbot = parallel_chain3.Chatbot()
        answer, sources, followup_qns = await chatbot.get_answer(query, chat_history)
    elif source == "Twitter":
        tweet_bot = twitter_bot.Tweets()
        answer, sources, followup_qns, augmentation = await tweet_bot.get_answer(query, chat_history)

        unique_data = []
        seen_entries = set()

        for entry in augmentation:
            content = entry[1][0]  # Primary content
            if content not in seen_entries:
                seen_entries.add(content)
                unique_data.append(entry)
 
    response['answer'] = answer

    await cache_instance.append_chat_history(uid, query, response['answer'])

    followup_questions = []
    if followup_qns:
        questions = followup_qns.get("questions", [])
        followup_questions.extend(questions)
    print("RESPONSE IN PROCESS QUERY",response)
    return response, sources, followup_questions, augmentation


def format_summary_enumerated(summary_enumerated):
    final_dict = {"teams": [], "projects": [], "members": [], "events": []}
    for item in summary_enumerated:
        # print(item)
        # print("item")
        # item[1][1]["content"] =  item[1][0]
        if "teams" in item[1][0]["source"]:
            final_dict["teams"] = final_dict["teams"] + [item[1][0]]
        elif "projects" in item[1][0]["source"]:
            final_dict["projects"] = final_dict["projects"] + [item[1][0]]
        elif "members" in item[1][0]["source"]:
            final_dict["members"] = final_dict["members"] + [item[1][0]]
        elif "events" in item[1][0]["source"]:
            final_dict["events"] = final_dict["events"] + [item[1][0]]
    return final_dict


# def process_query(uid,query):
#     response={}
#     Cache   = cache()
#     chat_history = Cache.retrieve_chat_history(uid)
#     chatbot = chain3.Chatbot()
#     chat_history = [{"role": "user", "content": ""}]
#     answer,sources,followup_qns = chatbot.get_answer(query, chat_history)
#     print("Answers-------------->",answer)
#     print("*"*120)
#     print("sources------------------------>",sources)
#     print("*"*120)
#     response['answer']=answer
#     Cache.append_chat_history(uid,query,response['answer'])
#     followup_questions=[]
#     if followup_qns:
#         # Access the 'questions' list within the 'Followup_Questions' dictionary
#         questions = followup_qns.get("questions", [])
#         followup_questions.append(questions)
#     return response,sources,followup_questions

class QueryRequest(BaseModel):
    """
    Represents the request payload for the /retrieve endpoint.

    Attributes:
        query (str): The query string for retrieval.
    """
    query: str
    UID: str
    source: str
    promptHistory: Optional[str] = None
    answerHistory: Optional[str] = None

    @validator("query")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        return value


class augumentationRequest(BaseModel):
    """
    Represents the request payload for the /retrieve endpoint.

    Attributes:
        query (str): The query string for retrieval.
    """
    query: str
    answer: str
    source: str
    references: Optional[list] = None

    @validator("query")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty or whitespace instead it has to be a string.")
        return value

    @validator("answer")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("answer cannot be empty or whitespace instead it has to be a string.")
        return value

    @validator("source")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("source cannot be empty or whitespace instead it has to be a string.")
        return value
    # @validator("references")
    # def query_must_not_be_empty(cls, value):
    #     if not value:
    #         raise ValueError("references cannot be empty it has to be a list of items.")
    #     return value


class FeedbackRequest(BaseModel):
    name: str
    email: str
    team: str
    directoryId: str
    prompt: str
    response: str
    rating: int
    comment: Optional[str] = None

    @validator("name")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("prompt cannot be empty or whitespace only.")
        return value

    @validator("email")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("prompt cannot be empty or whitespace only.")
        return value

    @validator("team")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("team cannot be empty or whitespace only.")
        return value

    @validator("directoryId")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("response cannot be empty or whitespace only.")
        return value

    @validator("prompt")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("prompt cannot be empty or whitespace only.")
        return value

    @validator("response")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("response cannot be empty or whitespace only.")
        return value

    @validator("rating")
    def query_must_not_be_empty(cls, value):
        if not isinstance(value, int):
            raise ValueError("Rating must be an integer.")
        return value


class FlushRequest(BaseModel):
    """
    Represents the request payload for the /retrieve endpoint.

    Attributes:
        query (str): The query string for retrieval.
    """
    UID: str

    @validator("UID")
    def query_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("UID cannot be empty or whitespace only.")
        return value




@app.get("/")
async def health():
    return {"Success": True}

@app.on_event("startup")
async def startup_event():
    # Initialize connection at the startup of the app
    storage_instance=Storage()
    storage_instance.connect_storage()


@app.post("/flush")
async def health(flush_request: FlushRequest):
    uid = flush_request.UID
    Cache = cache()
    flush_resp = Cache.flush_chat_history(uid)
    return {"Success": flush_resp}


@app.post("/augumented_info")
async def root(augumentationrequest: augumentationRequest):
    query = augumentationrequest.query
    answer = augumentationrequest.answer
    source = augumentationrequest.source
    references = augumentationrequest.references
    # reroute_instance = RerouteData()
    # summary_enumerated = reroute_instance.get_aug_info(answer,references)
    graph_retrieval_instance = graph_retrieval()
    search_terms_dict = graph_retrieval_instance.extract_entities(query, answer, references)
    print(search_terms_dict)

    search_terms = []
    if "person" in search_terms_dict:
        search_terms = search_terms + search_terms_dict["person"]
    if "organization" in search_terms_dict:
        search_terms = search_terms + search_terms_dict["organization"]
    if "Person" in search_terms_dict:
        search_terms = search_terms + search_terms_dict["Person"]
    if "Organization" in search_terms_dict:
        search_terms = search_terms + search_terms_dict["Organization"]
    # print(search_terms)
    summary_enumerated = graph_retrieval_instance.get_answer(search_terms)
    # summary_enumerated = format_summary_enumerated(summary_enumerated)
    return {"status": True, "augumented_info": summary_enumerated}


@app.post("/retrieve")
async def root(request: Request, query_request: QueryRequest):
    """
    Retrieves information based on the provided query.

    Args:
        query_request (QueryRequest): The request payload containing the query.

    Returns:
        dict: A dictionary containing the response.
    """
    try:
        # Ensure connection is reused
        query = query_request.query
        uid = query_request.UID
        source = query_request.source
        promptHistory = query_request.promptHistory
        answerHistory = query_request.answerHistory

        if query_request.promptHistory and query_request.answerHistory:
            Cache = cache()
            Cache.append_chat_history(uid, query_request.promptHistory, query_request.answerHistory)

        # Process the query and get the response tuple
        unpacked_response = await process_query(uid, query, source)
        print('----------unpacked response------------------------', unpacked_response)

        if "error" in unpacked_response:
            unpacked_response["status"] = False
            return unpacked_response
        else:
            # Unpack the response tuple
            response, source_list, followup_qns, references = unpacked_response
            conv_model = ConversationSchema(prompt=query, response=response)
            new_conversation = await add_conversation(conv_model)

            return {
                "status": True,
                "Query": query,
                "Response": response,
                "Source_list": source_list,
                "Followup_Questions": followup_qns,
                "references": references
            }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# from database import storage


# def check_db_connection():
#     db_con = storage().connect_storage()
#     try:
#         if "error" in db_con:
#             return {
#                 "status": False,
#                 "message": db_con["error"]
#             }
#     except:
#         return {
#             "status": True
#         }


@app.post("/feedback")
async def root(request: Request, feedback: FeedbackRequest):
    print(feedback)
    if (feedback.rating < 1) | (feedback.rating > 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    try:
        feedback_data = feedback.dict()
        new_feedback = await add_feedback(feedback)
        return new_feedback
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


"""import logging 
logger = logging.getLogger("uvicorn")
# Skip DEBUG and INFO logs
logger.setLevel(logging.WARNING)"""


# Function to check if the exact phrase appears in the text
def validate_Phrase_Labweek(text):
    phrase = "labweek field building"
    pattern = re.compile(r'\blabweek field building\b', re.IGNORECASE)
    return bool(pattern.search(text))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)#,workers=4)