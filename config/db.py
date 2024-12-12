import motor.motor_asyncio
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
load_dotenv()

MONGO_DETAILS = os.environ["MONGODB_URL"]

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.husky

feedback_collection = database.get_collection("feedback")
conversation_collection = database.get_collection("conversation")

def feedback_helper(feedback) -> dict:
    return {
        "id": str(feedback["_id"]),
        "prompt": feedback["prompt"],
        "response": feedback["response"],
        "rating": feedback["rating"],
        "comment" : feedback["comment"],
        "name" : feedback["name"],
        "team" : feedback["team"],
        "directoryId" : feedback["directoryId"],
        "email" : feedback["email"],
    }

def conversation_helper(conversation) -> dict:
    return {
        "id": str(conversation["_id"]),
        "prompt": conversation["prompt"],
        "response": conversation["response"],
        
    }
