from typing import Optional
from pydantic import BaseModel

class FeedbackModel(BaseModel):
    name: str
    team: str
    directoryId: str
    email: str
    prompt : str
    response : str
    rating : int
    comment : Optional[str] = None
