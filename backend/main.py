from dotenv import load_dotenv
load_dotenv()
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest, ChatResponse
from llm_providers import openai_provider, ollama_provider, claude_provider, gemini_provider

from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    provider: str
    prompt: str
    history: List[Message]
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "PolyChat backend is running!"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.provider == "openai":
        response = await openai_provider.generate_response(request.prompt, request.history)
    elif request.provider == "ollama":
        response = await ollama_provider.generate_response(request.prompt, request.history)
    elif request.provider == "claude":
        response = await claude_provider.generate_response(request.prompt, request.history)
    elif request.provider == "gemini":
        response = await gemini_provider.generate_response(request.prompt, request.history)
    else:
        raise HTTPException(status_code=400, detail="Unknown provider")
    return ChatResponse(response=response)

