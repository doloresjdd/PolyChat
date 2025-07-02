from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    provider: str  # "openai", "ollama", "claude"
    prompt: str
    history: Optional[List[str]] = []

class ChatResponse(BaseModel):
    response: str