# backend/api/routes/feedback.py
"""
反馈收集接口 - 收集用户对AI响应的反馈
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.shared import recommender, database, collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

# 请求模型
class FeedbackRequest(BaseModel):
    interaction_id: Optional[str] = None
    user_id: str
    provider: str
    query: Optional[str] = None
    satisfaction: float  # 0-1
    quality: Optional[float] = None  # 0-1
    helpful: Optional[bool] = None
    accurate: Optional[bool] = None
    fast: Optional[bool] = None
    comments: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str
    timestamp: str

@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    提交用户反馈
    
    Args:
        request: 反馈请求
        
    Returns:
        反馈提交结果
    """
    try:
        logger.info(f"📝 Feedback received: user={request.user_id}, provider={request.provider}, satisfaction={request.satisfaction:.2f}")
        
        # 1. 保存反馈到数据库
        if request.interaction_id:
            database.save_feedback(
                interaction_id=request.interaction_id,
                user_id=request.user_id,
                provider=request.provider,
                satisfaction=request.satisfaction,
                quality=request.quality,
                helpful=request.helpful,
                accurate=request.accurate,
                fast=request.fast,
                comments=request.comments
            )
        
        # 2. 收集反馈数据
        collector.collect_feedback(
            interaction_id=request.interaction_id or f"feedback_{datetime.now().timestamp()}",
            user_id=request.user_id,
            provider=request.provider,
            satisfaction=request.satisfaction,
            quality=request.quality,
            helpful=request.helpful,
            accurate=request.accurate,
            fast=request.fast,
            comments=request.comments
        )
        
        # 3. 更新推荐模型
        if request.query:
            recommender.update_from_feedback(
                provider=request.provider,
                query=request.query,
                user_id=request.user_id,
                feedback={
                    'satisfaction': request.satisfaction,
                    'quality': request.quality or request.satisfaction,
                    'helpful': request.helpful,
                    'accurate': request.accurate,
                    'fast': request.fast
                }
            )
        
        return FeedbackResponse(
            status="success",
            message="Feedback submitted successfully",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}")
async def get_user_feedback(user_id: str, limit: int = 50):
    """获取用户的反馈历史"""
    try:
        # 这里可以从数据库获取用户反馈
        # 暂时返回空列表
        return {
            "status": "success",
            "user_id": user_id,
            "feedback_history": [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting user feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_feedback_stats():
    """获取反馈统计信息"""
    try:
        collector_stats = collector.get_statistics()
        
        return {
            "status": "success",
            "statistics": collector_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

