# crud/feedback_crud.py
from typing import List, Union
from bson.objectid import ObjectId
from config.db import feedback_collection, feedback_helper
from models.feedback import FeedbackModel

# async def retrieve_feedbacks() -> List[dict]:
#     feedbacks = []
#     async for feedback in feedback_collection.find():
#         feedbacks.append(feedback_helper(feedback))
#     return feedbacks

async def add_feedback(feedback_data: FeedbackModel) -> dict:
    feedback = await feedback_collection.insert_one(feedback_data.dict())
    new_feedback = await feedback_collection.find_one({"_id": feedback.inserted_id})
    return feedback_helper(new_feedback)
