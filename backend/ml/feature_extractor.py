# backend/ml/feature_extractor.py
"""
特征提取模块 - 从查询和用户行为中提取特征
"""
import re
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        """初始化特征提取器"""
        try:
            # 加载spaCy模型
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using fallback")
            self.nlp = None
        
        # 加载句子嵌入模型
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"⚠️ Sentence transformer not available: {e}")
            self.embedding_model = None
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 查询类型关键词
        self.query_type_keywords = {
            'code': ['code', 'function', 'debug', 'error', 'python', 'javascript', 
                    'program', 'algorithm', 'implement', 'class', 'method', 'api'],
            'creative': ['write', 'story', 'poem', 'creative', 'imagine', 'describe',
                        'narrative', 'fiction', 'prose'],
            'analytical': ['analyze', 'data', 'statistics', 'calculate', 'graph', 
                          'chart', 'compare', 'evaluate', 'research'],
            'simple': ['what is', 'define', 'meaning', 'who is', 'when', 'where', 'explain'],
            'technical': ['how to', 'tutorial', 'guide', 'setup', 'install', 'configure'],
            'conversational': ['hello', 'hi', 'thanks', 'thank you', 'help', 'please']
        }
        
        logger.info("🚀 Feature extractor initialized")
    
    def extract_query_features(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        从查询中提取特征
        
        Args:
            query: 用户查询文本
            user_id: 用户ID（可选）
            
        Returns:
            特征字典
        """
        try:
            features = {
                'query_length': len(query),
                'word_count': len(query.split()),
                'char_count': len(query),
                'has_question_mark': '?' in query,
                'has_exclamation': '!' in query,
                'has_code': self._has_code(query),
                'has_url': bool(re.search(r'http[s]?://', query)),
                'has_email': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)),
                'has_number': bool(re.search(r'\d+', query)),
                'query_type': self._classify_query_type(query),
                'complexity_score': self._calculate_complexity(query),
                'language': self._detect_language(query),
                'sentiment': self._detect_sentiment(query),
                'entities': self._extract_entities(query),
                'keywords': self._extract_keywords(query),
                'embedding': self._get_embedding(query),
                'timestamp': datetime.now().isoformat()
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting query features: {e}")
            return self._get_default_features()
    
    def extract_user_features(
        self, 
        user_id: str, 
        user_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        从用户历史中提取特征
        
        Args:
            user_id: 用户ID
            user_history: 用户历史交互记录
            
        Returns:
            用户特征字典
        """
        try:
            if not user_history:
                return self._get_default_user_features(user_id)
            
            # 统计用户偏好
            provider_usage = {}
            query_types = {}
            avg_response_time = {}
            satisfaction_scores = {}
            
            for record in user_history:
                provider = record.get('provider', 'unknown')
                query_type = record.get('query_type', 'general')
                response_time = record.get('response_time', 0)
                satisfaction = record.get('satisfaction', 0.5)
                
                provider_usage[provider] = provider_usage.get(provider, 0) + 1
                query_types[query_type] = query_types.get(query_type, 0) + 1
                
                if provider not in avg_response_time:
                    avg_response_time[provider] = []
                avg_response_time[provider].append(response_time)
                
                if provider not in satisfaction_scores:
                    satisfaction_scores[provider] = []
                satisfaction_scores[provider].append(satisfaction)
            
            # 计算平均满意度
            avg_satisfaction = {}
            for provider, scores in satisfaction_scores.items():
                avg_satisfaction[provider] = np.mean(scores) if scores else 0.5
            
            features = {
                'user_id': user_id,
                'total_queries': len(user_history),
                'provider_preferences': provider_usage,
                'preferred_provider': max(provider_usage.items(), key=lambda x: x[1])[0] if provider_usage else None,
                'query_type_distribution': query_types,
                'avg_response_times': {k: np.mean(v) for k, v in avg_response_time.items()},
                'avg_satisfaction_scores': avg_satisfaction,
                'most_satisfied_provider': max(avg_satisfaction.items(), key=lambda x: x[1])[0] if avg_satisfaction else None,
                'user_activity_level': self._calculate_activity_level(len(user_history))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting user features: {e}")
            return self._get_default_user_features(user_id)
    
    def extract_provider_features(
        self, 
        provider: str, 
        provider_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        从AI提供商历史中提取特征
        
        Args:
            provider: AI提供商名称
            provider_history: 该提供商的历史记录
            
        Returns:
            提供商特征字典
        """
        try:
            if not provider_history:
                return self._get_default_provider_features(provider)
            
            # 统计提供商性能
            success_count = sum(1 for r in provider_history if r.get('success', False))
            total_count = len(provider_history)
            success_rate = success_count / total_count if total_count > 0 else 0.5
            
            response_times = [r.get('response_time', 0) for r in provider_history]
            avg_response_time = np.mean(response_times) if response_times else 1.0
            
            quality_scores = [r.get('quality_score', 0.5) for r in provider_history]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5
            
            # 按查询类型统计性能
            performance_by_type = {}
            for record in provider_history:
                query_type = record.get('query_type', 'general')
                if query_type not in performance_by_type:
                    performance_by_type[query_type] = {
                        'count': 0,
                        'success': 0,
                        'avg_quality': []
                    }
                performance_by_type[query_type]['count'] += 1
                if record.get('success', False):
                    performance_by_type[query_type]['success'] += 1
                performance_by_type[query_type]['avg_quality'].append(record.get('quality_score', 0.5))
            
            # 计算各类型的平均质量
            for qtype in performance_by_type:
                scores = performance_by_type[qtype]['avg_quality']
                performance_by_type[qtype]['avg_quality'] = np.mean(scores) if scores else 0.5
                performance_by_type[qtype]['success_rate'] = (
                    performance_by_type[qtype]['success'] / performance_by_type[qtype]['count']
                    if performance_by_type[qtype]['count'] > 0 else 0.5
                )
            
            features = {
                'provider': provider,
                'total_requests': total_count,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'avg_quality_score': avg_quality,
                'performance_by_query_type': performance_by_type,
                'best_query_type': max(
                    performance_by_type.items(), 
                    key=lambda x: x[1]['avg_quality']
                )[0] if performance_by_type else None,
                'reliability_score': success_rate * avg_quality
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting provider features: {e}")
            return self._get_default_provider_features(provider)
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        for qtype, keywords in self.query_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return qtype
        
        return 'general'
    
    def _has_code(self, query: str) -> bool:
        """检查是否包含代码"""
        code_indicators = ['```', 'def ', 'function', 'class ', 'import ', 'const ', 'let ', 'var ']
        return any(indicator in query for indicator in code_indicators)
    
    def _calculate_complexity(self, query: str) -> float:
        """计算查询复杂度（0-1）"""
        score = 0.0
        
        # 长度因子
        word_count = len(query.split())
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.2
        else:
            score += 0.1
        
        # 结构因子
        if self._has_code(query):
            score += 0.3
        if '?' in query:
            score += 0.1
        if len(query.split('.')) > 3:
            score += 0.2
        
        # 技术术语因子
        technical_terms = ['algorithm', 'optimize', 'implement', 'analyze', 'evaluate']
        if any(term in query.lower() for term in technical_terms):
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_language(self, query: str) -> str:
        """检测语言（简化版）"""
        # 检查中文字符
        if re.search(r'[\u4e00-\u9fff]', query):
            return 'chinese'
        # 检查日文字符
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', query):
            return 'japanese'
        # 检查韩文字符
        if re.search(r'[\uac00-\ud7a3]', query):
            return 'korean'
        return 'english'
    
    def _detect_sentiment(self, query: str) -> str:
        """检测情感（简化版）"""
        positive_words = ['thanks', 'thank', 'great', 'good', 'excellent', 'awesome', 'love']
        negative_words = ['bad', 'wrong', 'error', 'fail', 'problem', 'issue', 'help']
        
        query_lower = query.lower()
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        return 'neutral'
    
    def _extract_entities(self, query: str) -> List[str]:
        """提取命名实体"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
            return entities
        except:
            return []
    
    def _extract_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        try:
            words = query.lower().split()
            # 移除停用词
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            return keywords[:top_n]
        except:
            return []
    
    def _get_embedding(self, query: str) -> Optional[np.ndarray]:
        """获取查询的嵌入向量"""
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(query)
            return embedding.tolist()  # 转换为列表以便JSON序列化
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation failed: {e}")
            return None
    
    def _calculate_activity_level(self, total_queries: int) -> str:
        """计算用户活跃度"""
        if total_queries > 100:
            return 'high'
        elif total_queries > 20:
            return 'medium'
        else:
            return 'low'
    
    def _get_default_features(self) -> Dict[str, Any]:
        """获取默认查询特征"""
        return {
            'query_length': 0,
            'word_count': 0,
            'query_type': 'general',
            'complexity_score': 0.5,
            'language': 'english',
            'sentiment': 'neutral',
            'has_code': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_user_features(self, user_id: str) -> Dict[str, Any]:
        """获取默认用户特征"""
        return {
            'user_id': user_id,
            'total_queries': 0,
            'provider_preferences': {},
            'preferred_provider': None,
            'user_activity_level': 'low'
        }
    
    def _get_default_provider_features(self, provider: str) -> Dict[str, Any]:
        """获取默认提供商特征"""
        return {
            'provider': provider,
            'total_requests': 0,
            'success_rate': 0.5,
            'avg_response_time': 1.0,
            'avg_quality_score': 0.5,
            'reliability_score': 0.25
        }

