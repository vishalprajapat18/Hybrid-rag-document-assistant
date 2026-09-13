from pydantic import BaseModel
from typing import List
class CustomerFeedback(BaseModel):
    sentiment: str
    category: str
    summary: str

class Message(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    question: str
    history:List[Message]=[]


class ChatResponse(BaseModel):
    answer: str
   

