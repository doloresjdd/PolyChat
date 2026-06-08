# backend/main.py
"""
PolyChat AI - FastAPI主应用
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
from datetime import datetime

# 导入ML系统
from ml_services.ml_integration import ml_system

# 导入新的推荐系统
from api.routes import recommend, feedback
from api.shared import database as rec_database

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PolyChat AI - Intelligent Multi-AI Orchestration System",
    description="智能AI协调系统，具备学习、优化和个性化能力",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册推荐和反馈路由
app.include_router(recommend.router)
app.include_router(feedback.router)

# 数据模型
class ChatRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    history: Optional[List[Dict]] = None
    use_cache: bool = True
    providers: Optional[List[str]] = None

class ChatResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    message: str

# 健康检查
@app.get("/health")
async def health_check():
    """系统健康检查"""
    try:
        ml_health = ml_system.health_check()
        return {
            "status": "healthy",
            "ml_system": ml_health,
            "timestamp": ml_system.get_system_stats()['timestamp'] if 'timestamp' in ml_system.get_system_stats() else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 主要聊天接口
@app.post("/chat/enhanced", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest):
    """增强版聊天接口"""
    try:
        logger.info(f"Processing enhanced chat request: {request.prompt[:50]}...")
        
        # 调用ML系统处理查询
        result = await ml_system.process_query(
            prompt=request.prompt,
            user_id=request.user_id,
            history=request.history,
            use_cache=request.use_cache,
            providers=request.providers
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return ChatResponse(
            status="success",
            data=result,
            message="Query processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 系统统计接口
@app.get("/stats")
async def get_system_stats():
    """获取系统统计信息"""
    try:
        stats = ml_system.get_system_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 缓存管理接口
@app.post("/cache/clear")
async def clear_cache():
    """清空缓存"""
    try:
        # 这里需要实现缓存清空逻辑
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("�� PolyChat AI System starting up...")
    
    # 检查ML系统健康状态
    health = ml_system.health_check()
    if not health['overall']:
        logger.error("❌ ML system health check failed")
    else:
        logger.info("✅ ML system initialized successfully")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("�� PolyChat AI System shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# backend/main.py
# 在现有代码后添加以下端点

# 用户相关数据模型
class UserCreate(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    created_at: str

# 用户管理端点
@app.post("/api/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    """创建新用户（注册）"""
    try:
        logger.info(f"Creating user: {user.email}")
        
        # 这里应该实现真实的用户创建逻辑
        # 暂时返回模拟响应
        return UserResponse(
            id="user_001",
            email=user.email,
            name=user.name or user.email.split('@')[0],
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/login")
async def user_login(user: UserCreate):
    """用户登录"""
    try:
        logger.info(f"User login attempt: {user.email}")
        
        # 这里应该实现真实的登录验证
        # 暂时返回模拟响应
        return {
            "status": "success",
            "token": "mock_jwt_token_12345",
            "user": {
                "id": "user_001",
                "email": user.email,
                "name": user.email.split('@')[0]
            }
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/me")
async def get_current_user():
    """获取当前用户信息"""
    try:
        # 这里应该从JWT token中获取用户信息
        return {
            "id": "user_001",
            "email": "user@example.com",
            "name": "User"
        }
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 智能推荐端点（使用新的推荐系统）
@app.post("/api/chat/smart-recommend")
async def smart_recommend_chat(message_data: dict):
    """智能推荐AI聊天 - 使用新的推荐系统"""
    try:
        logger.info(f"Smart recommend request: {message_data.get('message', '')[:50]}...")
        
        # 1. 获取推荐
        from api.routes.recommend import get_recommendation
        from api.routes.recommend import RecommendRequest
        
        recommend_request = RecommendRequest(
            query=message_data.get('message', ''),
            user_id=message_data.get('user_id'),
            available_providers=['openai', 'claude', 'gemini', 'ollama']
        )
        
        recommendation_result = await get_recommendation(recommend_request)
        recommended_provider = recommendation_result.recommended_provider
        
        # 2. 调用推荐的AI提供商
        result = await ml_system.process_query(
            prompt=message_data.get('message', ''),
            user_id=message_data.get('user_id'),
            use_cache=True,
            providers=[recommended_provider]
        )
        
        if result.get('status') == 'success':
            # 提取AI回复
            responses = result.get('responses', {})
            if isinstance(responses, dict):
                ai_response = responses.get(recommended_provider, {}).get('response', '')
                if not ai_response:
                    first_provider = list(responses.keys())[0] if responses else None
                    if first_provider:
                        ai_response = responses[first_provider].get('response', '') if isinstance(responses[first_provider], dict) else str(responses[first_provider])
            else:
                ai_response = str(responses) if responses else "No response available"

            # 保存本次交互记录，用于后续反馈关联
            import uuid
            interaction_id = str(uuid.uuid4())
            try:
                rec_database.save_interaction(
                    interaction_id=interaction_id,
                    user_id=message_data.get('user_id', 'anonymous'),
                    query=message_data.get('message', ''),
                    provider=recommended_provider,
                    response=ai_response,
                    query_type=recommendation_result.query_type,
                    recommendation_id=None
                )
            except Exception as db_err:
                logger.warning(f"Failed to save interaction: {db_err}")

            # 返回推荐信息（含 interaction_id 供前端反馈使用）
            return {
                "status": "success",
                "message": ai_response,
                "response": ai_response,
                "recommended_provider": recommended_provider,
                "recommendation_reason": recommendation_result.recommendation_reason,
                "interaction_id": interaction_id,
                "ml_analysis": {
                    "query_type": recommendation_result.query_type,
                    "quality_score": recommendation_result.recommendation_score,
                    "processing_time": result.get('process_time', 0),
                    "cache_hit": result.get('cache_hit', False),
                    "confidence": recommendation_result.confidence,
                    "all_scores": recommendation_result.all_scores
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "error": "Smart recommendation failed",
                "provider": recommended_provider
            }
            
    except Exception as e:
        logger.error(f"Smart recommend error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": f"Smart recommendation error: {str(e)}",
            "provider": "smart-recommend"
        }

# 添加根端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "PolyChat AI - Intelligent Multi-AI Orchestration System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }