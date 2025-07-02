import os
import httpx

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def generate_response(prompt: str, history: list = None) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]