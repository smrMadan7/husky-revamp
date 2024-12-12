from typing import Optional
from pydantic import BaseModel

class ConversationModel(BaseModel):
    prompt: str
    response: dict
        
    