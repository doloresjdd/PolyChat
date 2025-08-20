# backend/main.py
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest, ChatResponse
from llm_providers import openai_provider, ollama_provider, claude_provider, gemini_provider
from pydantic import BaseModel
from typing import List, Optional, Dict

# 导入ML增强功能
from ml_integration import ml_system

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    provider: str
    prompt: str
    history: List[Message]

class EnhancedChatRequest(BaseModel):
    """增强版聊天请求"""
    prompt: str
    history: Optional[List[Message]] = []
    use_cache: Optional[bool] = True
    providers: Optional[List[str]] = None  # 如果为None，自动选择
    user_id: Optional[str] = None

class SingleProviderRequest(BaseModel):
    """单provider请求（向后兼容）"""
    provider: str
    prompt: str
    history: Optional[List[Message]] = []
    evaluate: Optional[bool] = False  # 是否评分

app = FastAPI(title="PolyChat API with ML Enhancement")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "PolyChat backend is running!",
        "ml_enabled": True,
        "version": "2.0"
    }

# 保留原有的端点（向后兼容）
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """原始聊天端点 - 向后兼容"""
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

# 新增ML增强端点
@app.post("/chat/enhanced")
async def enhanced_chat(request: EnhancedChatRequest):
    """
    ML增强版聊天端点
    - 自动选择最佳provider
    - 智能缓存
    - 响应评分
    - 多provider比较
    """
    try:
        result = await ml_system.process_enhanced_query(
            prompt=request.prompt,
            history=request.history,
            use_cache=request.use_cache,
            providers_to_use=request.providers
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/single")
async def single_provider_chat(request: SingleProviderRequest):
    """
    单provider聊天（带可选评分）
    """
    try:
        result = await ml_system.process_single_provider(
            provider=request.provider,
            prompt=request.prompt,
            history=request.history
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/compare")
async def compare_providers(request: EnhancedChatRequest):
    """
    比较多个providers的响应
    """
    try:
        # 强制使用所有provider进行比较
        all_providers = ['openai', 'claude', 'gemini', 'ollama']
        
        result = await ml_system.process_enhanced_query(
            prompt=request.prompt,
            history=request.history,
            use_cache=False,  # 比较时不使用缓存
            providers_to_use=request.providers or all_providers
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ML相关端点
@app.get("/stats")
async def get_statistics():
    """获取系统统计信息"""
    return ml_system.get_statistics()

@app.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计"""
    return ml_system.cache.get_cache_stats()

@app.get("/api/performance")
async def get_api_performance():
    """获取API性能统计"""
    return ml_system.optimizer.get_api_stats()

@app.post("/cache/search")
async def search_cache(query: str):
    """搜索缓存中的相似查询"""
    result = ml_system.cache.search_similar(query)
    return result if result else {"message": "No similar queries found"}

@app.post("/classify")
async def classify_query(query: str):
    """分类查询类型"""
    query_type = ml_system.optimizer.classify_query(query)
    
    # 简单的推荐逻辑
    if query_type == 'code':
        recommended = ['openai', 'claude']
    elif query_type == 'creative':
        recommended = ['claude', 'openai']
    elif query_type == 'simple':
        recommended = ['gemini', 'ollama']
    else:
        recommended = ['openai']
    
    return {
        "query": query,
        "type": query_type,
        "recommended_providers": recommended
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "ml_components": {
            "scorer": "ready",
            "cache": "ready",
            "optimizer": "ready"
        },
        "providers_available": list(ml_system.providers.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)