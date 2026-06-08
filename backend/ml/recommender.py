# backend/ml/recommender.py
"""
推荐模型 - 基于特征和用户偏好推荐最佳AI提供商
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import os
from datetime import datetime

try:
    from .feature_extractor import FeatureExtractor
    from .bandit import ThompsonSamplingBandit
except ImportError:
    # 处理相对导入问题
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ml.feature_extractor import FeatureExtractor
    from ml.bandit import ThompsonSamplingBandit

logger = logging.getLogger(__name__)

class AIRecommender:
    def __init__(self, model_path: Optional[str] = None):
        """初始化推荐模型"""
        self.feature_extractor = FeatureExtractor()
        self.bandit = ThompsonSamplingBandit()
        
        # 模型参数
        self.model_path = model_path or "./models/recommender_model.json"
        self.weights = self._load_weights()
        
        # 特征权重（可学习）
        self.feature_weights = {
            'query_type_match': 0.3,      # 查询类型匹配度
            'user_preference': 0.25,      # 用户历史偏好
            'provider_performance': 0.25,  # 提供商历史性能
            'response_time': 0.1,         # 响应时间
            'cost_efficiency': 0.1        # 成本效率
        }
        
        # 提供商成本（每1000 tokens）
        self.provider_costs = {
            'openai': 0.002,
            'claude': 0.008,
            'gemini': 0.001,
            'ollama': 0.0
        }
        
        logger.info("🚀 AI Recommender initialized")
    
    def recommend(
        self,
        query: str,
        user_id: Optional[str] = None,
        user_history: Optional[List[Dict]] = None,
        provider_history: Optional[Dict[str, List[Dict]]] = None,
        available_providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        推荐最佳AI提供商
        
        Args:
            query: 用户查询
            user_id: 用户ID
            user_history: 用户历史交互
            provider_history: 各提供商的历史记录
            available_providers: 可用的提供商列表
            
        Returns:
            推荐结果字典
        """
        try:
            # 默认可用提供商
            if available_providers is None:
                available_providers = ['openai', 'claude', 'gemini', 'ollama']
            
            # 1. 提取特征
            query_features = self.feature_extractor.extract_query_features(query, user_id)
            user_features = self.feature_extractor.extract_user_features(user_id, user_history)
            
            # 2. 为每个提供商计算推荐分数
            provider_scores = {}
            recommendation_reasons = {}
            
            for provider in available_providers:
                # 提取提供商特征
                provider_hist = provider_history.get(provider, []) if provider_history else []
                provider_features = self.feature_extractor.extract_provider_features(
                    provider, provider_hist
                )
                
                # 计算综合分数
                score, reasons = self._calculate_recommendation_score(
                    query_features,
                    user_features,
                    provider_features,
                    provider
                )
                
                provider_scores[provider] = score
                recommendation_reasons[provider] = reasons
            
            # 3. 使用Thompson Sampling进行探索-利用平衡（传入 query_type 以使用冷启动先验）
            recommended_provider = self.bandit.select_arm(
                provider_scores,
                available_providers,
                query_type=query_features.get('query_type', 'general')
            )
            
            # 4. 生成推荐理由
            recommendation_reason = self._generate_recommendation_reason(
                recommended_provider,
                provider_scores,
                recommendation_reasons,
                query_features
            )
            
            # 5. 返回推荐结果
            result = {
                'recommended_provider': recommended_provider,
                'recommendation_score': provider_scores[recommended_provider],
                'all_scores': provider_scores,
                'recommendation_reason': recommendation_reason,
                'confidence': self._calculate_confidence(provider_scores, recommended_provider),
                'query_type': query_features['query_type'],
                'alternative_providers': self._get_alternatives(provider_scores, recommended_provider),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Recommended: {recommended_provider} (score: {provider_scores[recommended_provider]:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in recommendation: {e}")
            # 返回默认推荐
            return {
                'recommended_provider': 'openai',
                'recommendation_score': 0.5,
                'recommendation_reason': 'Default recommendation due to error',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _calculate_recommendation_score(
        self,
        query_features: Dict,
        user_features: Dict,
        provider_features: Dict,
        provider: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        计算推荐分数
        
        Returns:
            (分数, 理由字典)
        """
        reasons = {}
        score = 0.0
        
        # 1. 查询类型匹配度
        query_type = query_features.get('query_type', 'general')
        provider_best_type = provider_features.get('best_query_type')
        
        if provider_best_type == query_type:
            type_match_score = 1.0
            reasons['query_type_match'] = f"Excellent match: {provider} performs best on {query_type} queries"
        elif provider_best_type:
            type_match_score = 0.5
            reasons['query_type_match'] = f"Partial match: {provider} is good for {provider_best_type} queries"
        else:
            type_match_score = 0.3
            reasons['query_type_match'] = f"General purpose: {provider} handles various query types"
        
        score += type_match_score * self.feature_weights['query_type_match']
        
        # 2. 用户历史偏好
        user_pref_score = 0.5  # 默认中性
        provider_prefs = user_features.get('provider_preferences', {})
        total_queries = user_features.get('total_queries', 0)
        
        if provider in provider_prefs and total_queries > 0:
            usage_ratio = provider_prefs[provider] / total_queries
            user_pref_score = min(1.0, usage_ratio * 2)  # 使用频率越高，分数越高
            reasons['user_preference'] = f"User has used {provider} {provider_prefs[provider]} times ({usage_ratio*100:.1f}% of queries)"
        else:
            reasons['user_preference'] = f"No previous usage of {provider} by this user"
        
        # 检查用户满意度
        avg_satisfaction = user_features.get('avg_satisfaction_scores', {}).get(provider, 0.5)
        if avg_satisfaction > 0.7:
            user_pref_score += 0.2
            reasons['user_satisfaction'] = f"User has high satisfaction ({avg_satisfaction:.2f}) with {provider}"
        elif avg_satisfaction < 0.3:
            user_pref_score -= 0.2
            reasons['user_satisfaction'] = f"User has low satisfaction ({avg_satisfaction:.2f}) with {provider}"
        
        user_pref_score = max(0.0, min(1.0, user_pref_score))
        score += user_pref_score * self.feature_weights['user_preference']
        
        # 3. 提供商历史性能
        success_rate = provider_features.get('success_rate', 0.5)
        avg_quality = provider_features.get('avg_quality_score', 0.5)
        reliability = provider_features.get('reliability_score', 0.25)
        
        performance_score = (success_rate * 0.4 + avg_quality * 0.4 + reliability * 0.2)
        score += performance_score * self.feature_weights['provider_performance']
        
        reasons['provider_performance'] = (
            f"Success rate: {success_rate:.2f}, "
            f"Avg quality: {avg_quality:.2f}, "
            f"Reliability: {reliability:.2f}"
        )
        
        # 4. 响应时间
        avg_response_time = provider_features.get('avg_response_time', 1.0)
        # 响应时间越短，分数越高（归一化到0-1）
        time_score = max(0.0, 1.0 - (avg_response_time / 5.0))  # 假设5秒为基准
        score += time_score * self.feature_weights['response_time']
        
        reasons['response_time'] = f"Average response time: {avg_response_time:.2f}s"
        
        # 5. 成本效率
        cost = self.provider_costs.get(provider, 0.002)
        # 成本越低，分数越高
        cost_score = max(0.0, 1.0 - (cost / 0.01))  # 假设0.01为最高成本
        score += cost_score * self.feature_weights['cost_efficiency']
        
        reasons['cost_efficiency'] = f"Cost per 1K tokens: ${cost:.4f}"
        
        # 确保分数在0-1范围内
        final_score = max(0.0, min(1.0, score))
        
        return final_score, reasons
    
    def _generate_recommendation_reason(
        self,
        provider: str,
        scores: Dict[str, float],
        reasons: Dict[str, Dict],
        query_features: Dict
    ) -> str:
        """生成推荐理由文本"""
        provider_reasons = reasons.get(provider, {})
        query_type = query_features.get('query_type', 'general')
        
        # 构建理由
        reason_parts = []
        
        # 主要理由
        if 'query_type_match' in provider_reasons:
            reason_parts.append(provider_reasons['query_type_match'])
        
        if 'user_preference' in provider_reasons:
            reason_parts.append(provider_reasons['user_preference'])
        
        if 'provider_performance' in provider_reasons:
            reason_parts.append(f"Historical performance: {provider_reasons['provider_performance']}")
        
        # 如果没有特定理由，使用通用理由
        if not reason_parts:
            reason_parts.append(
                f"{provider} is recommended based on overall performance and suitability for {query_type} queries"
            )
        
        return " | ".join(reason_parts[:2])  # 最多显示2个主要理由
    
    def _calculate_confidence(self, scores: Dict[str, float], recommended: str) -> float:
        """计算推荐置信度"""
        if not scores:
            return 0.5
        
        recommended_score = scores[recommended]
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return 0.5
        
        # 置信度基于推荐分数与最高分数的接近程度
        confidence = (recommended_score - min_score) / (max_score - min_score)
        
        # 如果推荐分数明显高于其他，置信度更高
        other_scores = [s for p, s in scores.items() if p != recommended]
        if other_scores:
            avg_other = np.mean(other_scores)
            if recommended_score > avg_other * 1.2:  # 高出20%以上
                confidence = min(1.0, confidence * 1.2)
        
        return max(0.0, min(1.0, confidence))
    
    def _get_alternatives(self, scores: Dict[str, float], recommended: str) -> List[Dict[str, Any]]:
        """获取备选提供商"""
        alternatives = []
        
        # 按分数排序
        sorted_providers = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 获取前3个备选（排除已推荐的）
        for provider, score in sorted_providers:
            if provider != recommended:
                alternatives.append({
                    'provider': provider,
                    'score': score,
                    'score_difference': scores[recommended] - score
                })
                if len(alternatives) >= 2:  # 最多2个备选
                    break
        
        return alternatives
    
    def update_from_feedback(
        self,
        provider: str,
        query: str,
        user_id: str,
        feedback: Dict[str, Any]
    ):
        """
        根据用户反馈更新模型
        
        Args:
            provider: 使用的提供商
            query: 查询文本
            user_id: 用户ID
            feedback: 反馈数据（包含satisfaction, quality等）
        """
        try:
            # 更新bandit
            satisfaction = feedback.get('satisfaction', 0.5)
            success = satisfaction > 0.6  # 满意度>0.6视为成功
            
            self.bandit.update(provider, success)
            
            # 可以在这里添加更复杂的模型更新逻辑
            # 例如：调整特征权重、更新性能统计等
            
            logger.info(f"✅ Updated recommendation model from feedback: {provider}, satisfaction: {satisfaction:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating from feedback: {e}")
    
    def _load_weights(self) -> Dict:
        """加载模型权重"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load weights: {e}")
        return {}
    
    def save_weights(self):
        """保存模型权重"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'w') as f:
                json.dump(self.weights, f, indent=2)
            logger.info("✅ Model weights saved")
        except Exception as e:
            logger.error(f"❌ Failed to save weights: {e}")

