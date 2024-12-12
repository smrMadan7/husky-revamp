# crud/conversation_crud.py
from typing import List, Union
from bson.objectid import ObjectId
from config.db import conversation_collection, conversation_helper
from models.conversation import ConversationModel


async def add_conversation(conversation_data: ConversationModel) -> dict:
    conversation = await conversation_collection.insert_one(conversation_data.dict())
    new_conversation = await conversation_collection.find_one({"_id": conversation.inserted_id})
    return conversation_helper(new_conversation)


#  conversation = await conversation_collection.insert_one(conversation_data)
#     # new_conversation = await conversation_collection.find_one({"_id": conversation.inserted_id})
#     return (conversation)
