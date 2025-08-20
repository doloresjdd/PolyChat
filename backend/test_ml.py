# backend/test_ml.py
import asyncio
import aiohttp
import json

async def test_enhanced_chat():
    """测试增强版聊天"""
    url = "http://localhost:8000/chat/enhanced"
    
    # 测试查询
    queries = [
        "Write a Python function to sort a list",
        "What is machine learning?",
        "Tell me a joke"
    ]
    
    async with aiohttp.ClientSession() as session:
        for query in queries:
            print(f"\n📝 Query: {query}")
            
            payload = {
                "prompt": query,
                "history": [],
                "use_cache": True
            }
            
            async with session.post(url, json=payload) as response:
                result = await response.json()
                
                print(f"✅ Cache Hit: {result.get('cache_hit', False)}")
                print(f"🎯 Query Type: {result.get('query_type', 'unknown')}")
                print(f"⭐ Best Provider: {result.get('recommended_provider', 'none')}")
                print(f"⏱️ Process Time: {result.get('process_time', 0):.2f}s")
                
                if 'responses' in result:
                    print("\n📊 Provider Scores:")
                    for provider, data in result['responses'].items():
                        if 'scores' in data:
                            score = data['scores']['overall']
                            print(f"  {provider}: {score:.2%}")

async def test_compare():
    """测试provider比较"""
    url = "http://localhost:8000/chat/compare"
    
    payload = {
        "prompt": "Explain quantum computing in simple terms",
        "history": []
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print("\n🔍 Comparison Results:")
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_enhanced_chat())
    asyncio.run(test_compare())