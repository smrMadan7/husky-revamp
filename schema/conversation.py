# schemas/Conversation.py
from pydantic import BaseModel
from typing import Optional

class ConversationSchema(BaseModel):
    prompt: str
    response: dict
   
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Hannah Howard",
                "response": "Hannah Howard is identified as a Starfleet Software Engineer within the Engineering team, focusing on Blockchain Infrastructure at Protocol Labs. Her role involves contributing to the development and maintenance of blockchain technologies, which are crucial for the secure and efficient operation of decentralized networks",
            }
        }
