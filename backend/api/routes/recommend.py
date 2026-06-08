# backend/api/routes/recommend.py
"""
推荐接口 - 智能推荐最佳AI提供商
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.shared import recommender, database, collector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommend", tags=["recommendation"])

# 请求模型
class RecommendRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    available_providers: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

class RecommendResponse(BaseModel):
    status: str
    recommended_provider: str
    recommendation_score: float
    recommendation_reason: str
    confidence: float
    query_type: str
    all_scores: Dict[str, float]
    alternative_providers: List[Dict[str, Any]]
    timestamp: str

@router.post("/", response_model=RecommendResponse)
async def get_recommendation(request: RecommendRequest):
    """
    获取AI提供商推荐
    
    Args:
        request: 推荐请求
        
    Returns:
        推荐结果
    """
    try:
        logger.info(f"📝 Recommendation request: user={request.user_id}, query={request.query[:50]}...")
        
        # 1. 获取用户历史
        user_history = []
        provider_history = {}
        
        if request.user_id:
            # 从数据库获取用户历史
            user_interactions = database.get_user_history(request.user_id, limit=100)
            
            # 转换为推荐系统需要的格式
            for interaction in user_interactions:
                user_history.append({
                    'provider': interaction.get('provider'),
                    'query_type': interaction.get('query_type', 'general'),
                    'response_time': 1.0,  # 可以从metadata中获取
                    'satisfaction': 0.5,   # 可以从feedback中获取
                    'timestamp': interaction.get('timestamp')
                })
            
            # 获取各提供商的历史
            for provider in ['openai', 'claude', 'gemini', 'ollama']:
                provider_interactions = database.get_provider_history(provider, limit=100)
                provider_history[provider] = [
                    {
                        'success': True,  # 可以从feedback中判断
                        'response_time': 1.0,
                        'quality_score': 0.5,
                        'query_type': i.get('query_type', 'general')
                    }
                    for i in provider_interactions
                ]
        
        # 2. 获取推荐
        recommendation = recommender.recommend(
            query=request.query,
            user_id=request.user_id,
            user_history=user_history if user_history else None,
            provider_history=provider_history if provider_history else None,
            available_providers=request.available_providers
        )
        
        # 3. 保存推荐记录
        if request.user_id:
            recommendation_id = database.save_recommendation(
                user_id=request.user_id,
                query=request.query,
                recommended_provider=recommendation['recommended_provider'],
                recommendation_score=recommendation['recommendation_score'],
                recommendation_reason=recommendation['recommendation_reason'],
                query_type=recommendation['query_type']
            )
            
            # 收集推荐数据
            collector.collect_recommendation_result(
                query=request.query,
                user_id=request.user_id,
                recommended_provider=recommendation['recommended_provider'],
                recommendation_score=recommendation['recommendation_score']
            )
        
        # 4. 返回推荐结果（将 numpy float 转为 Python float 避免序列化错误）
        return RecommendResponse(
            status="success",
            recommended_provider=recommendation['recommended_provider'],
            recommendation_score=float(recommendation['recommendation_score']),
            recommendation_reason=recommendation['recommendation_reason'],
            confidence=float(recommendation['confidence']),
            query_type=recommendation['query_type'],
            all_scores={k: float(v) for k, v in recommendation['all_scores'].items()},
            alternative_providers=[
                {**p, 'score': float(p['score']), 'score_difference': float(p['score_difference'])}
                for p in recommendation.get('alternative_providers', [])
            ],
            timestamp=recommendation['timestamp']
        )
        
    except Exception as e:
        logger.error(f"❌ Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_recommendation_stats():
    """获取推荐系统统计信息"""
    try:
        bandit_stats = recommender.bandit.get_statistics()
        collector_stats = collector.get_statistics()
        
        return {
            "status": "success",
            "bandit_statistics": bandit_stats,
            "collector_statistics": collector_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

