# schemas/Feedback.py
from pydantic import BaseModel
from typing import Optional

class FeedbackSchema(BaseModel):
     
    name: str
    team: str
    directoryId: str
    email: str
    prompt : str
    response : str
    rating : int
    comment : Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "name" : "Hannah Howard",
                "team" : "pllabs",
                "directoryId" : "cldvodkwy06tfu21kxuc7i639",
                "email" : "hannah@pllabs.xyz",
                "prompt": "Tell me about Hannah Howard",
                "response": "Hannah Howard is identified as a Starfleet Software Engineer within the Engineering team, focusing on Blockchain Infrastructure at Protocol Labs. Her role involves contributing to the development and maintenance of blockchain technologies, which are crucial for the secure and efficient operation of decentralized networks",
                "rating" : 5,
                "comment" : "Accurate and helpful"
            }
        }
