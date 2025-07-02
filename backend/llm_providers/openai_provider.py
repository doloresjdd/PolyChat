import os
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    
async def generate_response(prompt: str, history: list = None) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    messages = (history if history else []) + [{"role": "user", "content": prompt}]
    data = {
    "model": "gpt-3.5-turbo",
    "messages": [msg.dict() for msg in history] + [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]