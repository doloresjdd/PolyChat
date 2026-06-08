# backend/main_simple.py
"""
PolyChat AI - 简化版FastAPI应用（仅登录功能）
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PolyChat AI - Simple Login Server",
    description="简化的登录服务器",
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

# 数据模型
class UserCreate(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    name: Optional[str] = None
    created_at: str

# 健康检查
@app.get("/health")
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Simple login server is running"
    }

# 用户登录
@app.post("/api/users/login")
async def user_login(user: UserCreate):
    """用户登录"""
    try:
        logger.info(f"🔐 User login attempt: {user.email}")
        logger.info(f"📝 Request data: {user.dict()}")
        
        # 简单的登录验证（任何邮箱密码都通过）
        response_data = {
            "status": "success",
            "token": "mock_jwt_token_12345",
            "user": {
                "id": "user_001",
                "email": user.email,
                "name": user.email.split('@')[0]
            }
        }
        
        logger.info(f"✅ Login successful for {user.email}")
        logger.info(f"📤 Sending response: {response_data}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 用户注册
@app.post("/api/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    """创建新用户（注册）"""
    try:
        logger.info(f"Creating user: {user.email}")
        
        return UserResponse(
            id="user_001",
            email=user.email,
            name=user.name or user.email.split('@')[0],
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 获取当前用户信息
@app.get("/api/users/me")
async def get_current_user():
    """获取当前用户信息"""
    try:
        return {
            "id": "user_001",
            "email": "user@example.com",
            "name": "User"
        }
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "PolyChat AI - Simple Login Server",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
