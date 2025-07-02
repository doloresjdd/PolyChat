import os
import httpx

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

async def generate_response(prompt: str, history: list = None) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    model = "claude-3-7-sonnet-20250219"
    messages = [msg.dict() for msg in history] if history else []
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model,
        "max_tokens": 1024,
        "messages": messages
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["content"][0]["text"]