import os
import httpx

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

async def generate_response(prompt: str, history: list = None) -> str:
    url = f"{OLLAMA_API_URL}/api/generate"
    data = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]