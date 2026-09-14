from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = []
    document_id: Optional[str] = None   # limit retrieval to one uploaded document

class Source(BaseModel):
    filename: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []
