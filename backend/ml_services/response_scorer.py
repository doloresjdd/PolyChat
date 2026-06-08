# backend/ml_services/response_scorer.py
"""
多维度响应评分系统 - 基于NLP的智能评估
"""
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ResponseScorer:
    def __init__(self):
        """初始化响应评分系统"""
        try:
            # 加载spaCy模型（英文）
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except OSError:
            logger.warning("⚠️ spaCy model not found, installing...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model installed and loaded")
            except Exception as e:
                logger.error(f"❌ Failed to install spaCy model: {e}")
                self.nlp = None
        
        # 初始化TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 评分权重配置
        self.scoring_weights = {
            'relevance': 0.25,      # 相关性
            'completeness': 0.20,   # 完整性
            'clarity': 0.20,        # 清晰度
            'accuracy': 0.15,       # 准确性
            'usefulness': 0.10,     # 实用性
            'structure': 0.10       # 结构化
        }
        
        logger.info("🚀 Response scorer initialized successfully")
    
    async def score_response(
        self, 
        query: str, 
        response: str, 
        provider: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        对AI响应进行多维度评分
        
        Args:
            query: 用户查询
            response: AI响应
            provider: AI提供商
            context: 额外上下文信息
            
        Returns:
            评分结果字典
        """
        try:
            logger.info(f"📊 Scoring response from {provider}...")
            
            # 基础文本分析
            doc = self.nlp(response) if self.nlp else None
            
            # 计算各维度分数
            scores = {
                'relevance': self._calculate_relevance(query, response),
                'completeness': self._calculate_completeness(query, response, doc),
                'clarity': self._calculate_clarity(response, doc),
                'accuracy': self._calculate_accuracy(response, doc),
                'usefulness': self._calculate_usefulness(response, doc),
                'structure': self._calculate_structure(response)
            }
            
            # 计算加权总分
            total_score = sum(
                scores[dim] * self.scoring_weights[dim] 
                for dim in scores.keys()
            )
            
            # 添加特殊奖励分
            bonus_points = self._calculate_bonus_points(response, query)
            total_score += bonus_points
            
            # 确保分数在0-100范围内
            total_score = max(0, min(100, total_score))
            
            # 生成详细报告
            detailed_report = self._generate_detailed_report(scores, bonus_points, total_score)
            
            result = {
                'total_score': round(total_score, 2),
                'dimension_scores': {k: round(v, 2) for k, v in scores.items()},
                'bonus_points': round(bonus_points, 2),
                'detailed_report': detailed_report,
                'provider': provider,
                'timestamp': self._get_timestamp()
            }
            
            logger.info(f"✅ Response scored: {total_score:.2f}/100")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error scoring response: {e}")
            return {
                'total_score': 0,
                'error': str(e),
                'provider': provider
            }
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """计算响应与查询的相关性"""
        try:
            # 使用TF-IDF计算相似度
            texts = [query, response]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # 转换为0-100分数
            return min(100, similarity * 100)
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 50.0
    
    def _calculate_completeness(self, query: str, response: str, doc) -> float:
        """计算响应的完整性"""
        try:
            score = 50.0  # 基础分
            
            # 检查是否回答了问题类型
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            query_lower = query.lower()
            
            if any(word in query_lower for word in question_words):
                if 'what' in query_lower and len(response.split()) > 20:
                    score += 20
                if 'how' in query_lower and ('step' in response.lower() or 'process' in response.lower()):
                    score += 20
                if 'why' in query_lower and ('because' in response.lower() or 'reason' in response.lower()):
                    score += 20
            
            # 检查响应长度是否合适
            word_count = len(response.split())
            if 50 <= word_count <= 300:
                score += 15
            elif 300 < word_count <= 600:
                score += 10
            
            # 检查是否包含具体信息
            if re.search(r'\d+', response):  # 包含数字
                score += 10
            if re.search(r'[A-Z][a-z]+', response):  # 包含专有名词
                score += 5
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 50.0
    
    def _calculate_clarity(self, response: str, doc) -> float:
        """计算响应的清晰度"""
        try:
            score = 50.0
            
            if doc:
                # 使用spaCy分析句子结构
                sentences = list(doc.sents)
                if len(sentences) > 1:
                    score += 10
                
                # 检查句子长度
                avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
                if 10 <= avg_sentence_length <= 25:
                    score += 15
                elif 25 < avg_sentence_length <= 35:
                    score += 10
            
            # 检查标点符号使用
            if response.count('.') > 0:
                score += 10
            
            # 检查段落结构
            if '\n\n' in response:
                score += 10
            
            # 检查是否有明确的结论
            conclusion_words = ['therefore', 'thus', 'conclusion', 'summary', 'in summary']
            if any(word in response.lower() for word in conclusion_words):
                score += 5
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating clarity: {e}")
            return 50.0
    
    def _calculate_accuracy(self, response: str, doc) -> float:
        """计算响应的准确性（基于文本特征）"""
        try:
            score = 50.0
            
            # 检查是否包含引用或来源
            if re.search(r'http[s]?://', response):
                score += 15
            
            # 检查是否包含具体数据
            if re.search(r'\d+\.\d+%', response) or re.search(r'\d+%', response):
                score += 10
            
            # 检查是否包含年份
            if re.search(r'\b(19|20)\d{2}\b', response):
                score += 10
            
            # 检查是否包含专业术语
            if doc:
                entities = [ent.label_ for ent in doc.ents]
                if 'PERSON' in entities:
                    score += 5
                if 'ORG' in entities:
                    score += 5
                if 'GPE' in entities:
                    score += 5
            
            # 检查是否包含代码块
            if '```' in response:
                score += 10
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 50.0
    
    def _calculate_usefulness(self, response: str, doc) -> float:
        """计算响应的实用性"""
        try:
            score = 50.0
            
            # 检查是否包含可操作的建议
            action_words = ['should', 'can', 'will', 'must', 'need to', 'try to']
            if any(word in response.lower() for word in action_words):
                score += 15
            
            # 检查是否包含列表或步骤
            if re.search(r'\d+\.', response) or re.search(r'•', response):
                score += 15
            
            # 检查是否包含示例
            example_words = ['example', 'instance', 'such as', 'like', 'e.g.']
            if any(word in response.lower() for word in example_words):
                score += 10
            
            # 检查情感极性（使用TextBlob）
            try:
                blob = TextBlob(response)
                polarity = blob.sentiment.polarity
                if -0.1 <= polarity <= 0.1:  # 中性情感通常更客观
                    score += 10
            except:
                pass
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating usefulness: {e}")
            return 50.0
    
    def _calculate_structure(self, response: str) -> float:
        """计算响应的结构化程度"""
        try:
            score = 50.0
            
            # 检查标题结构
            if re.search(r'^#+\s+', response, re.MULTILINE):
                score += 20
            
            # 检查列表结构
            if re.search(r'^\s*[-*•]\s+', response, re.MULTILINE):
                score += 15
            
            # 检查编号列表
            if re.search(r'^\s*\d+\.\s+', response, re.MULTILINE):
                score += 15
            
            # 检查表格结构
            if '|' in response and '-' in response:
                score += 10
            
            # 检查代码块
            if '```' in response:
                score += 10
            
            # 检查段落分隔
            if response.count('\n\n') >= 2:
                score += 10
            
            return min(100, score)
            
        except Exception as e:
            logger.error(f"Error calculating structure: {e}")
            return 50.0
    
    def _calculate_bonus_points(self, response: str, query: str) -> float:
        """计算特殊奖励分"""
        bonus = 0.0
        
        # 代码相关奖励
        if '```' in response:
            if 'python' in response.lower() or 'javascript' in response.lower():
                bonus += 5
            if '```' in response and response.count('```') >= 2:
                bonus += 3
        
        # 数学公式奖励
        if re.search(r'\$.*\$', response) or '=' in response:
            bonus += 3
        
        # 链接奖励
        if re.search(r'http[s]?://', response):
            bonus += 2
        
        # 图片引用奖励
        if '![alt]' in response or 'image' in response.lower():
            bonus += 2
        
        # 多语言支持奖励
        if re.search(r'[а-яё]', response) or re.search(r'[一-龯]', response):
            bonus += 3
        
        return min(20, bonus)  # 最大奖励20分
    
    def _generate_detailed_report(
        self, 
        scores: Dict[str, float], 
        bonus_points: float, 
        total_score: float
    ) -> Dict[str, Any]:
        """生成详细的评分报告"""
        try:
            # 找出最强和最弱的维度
            strongest_dim = max(scores.items(), key=lambda x: x[1])
            weakest_dim = min(scores.items(), key=lambda x: x[1])
            
            # 生成改进建议
            suggestions = self._generate_suggestions(scores)
            
            return {
                'strongest_dimension': {
                    'name': strongest_dim[0],
                    'score': strongest_dim[1],
                    'description': self._get_dimension_description(strongest_dim[0])
                },
                'weakest_dimension': {
                    'name': weakest_dim[0],
                    'score': weakest_dim[1],
                    'description': self._get_dimension_description(weakest_dim[0])
                },
                'improvement_suggestions': suggestions,
                'score_breakdown': {
                    'base_score': round(total_score - bonus_points, 2),
                    'bonus_points': round(bonus_points, 2),
                    'final_score': round(total_score, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            return {}
    
    def _generate_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if scores['relevance'] < 70:
            suggestions.append("Focus more on directly addressing the user's question")
        
        if scores['completeness'] < 70:
            suggestions.append("Provide more comprehensive information and examples")
        
        if scores['clarity'] < 70:
            suggestions.append("Use shorter sentences and clearer language")
        
        if scores['accuracy'] < 70:
            suggestions.append("Include more specific facts, data, or citations")
        
        if scores['usefulness'] < 70:
            suggestions.append("Add actionable steps or practical examples")
        
        if scores['structure'] < 70:
            suggestions.append("Use better formatting with headers, lists, and paragraphs")
        
        return suggestions[:3]  # 最多3个建议
    
    def _get_dimension_description(self, dimension: str) -> str:
        """获取维度描述"""
        descriptions = {
            'relevance': 'How well the response addresses the user\'s question',
            'completeness': 'How comprehensive and thorough the response is',
            'clarity': 'How easy the response is to understand',
            'accuracy': 'How factual and reliable the information is',
            'usefulness': 'How actionable and practical the response is',
            'structure': 'How well-organized and formatted the response is'
        }
        return descriptions.get(dimension, 'Unknown dimension')
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def compare_responses(
        self, 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """比较多个响应的质量"""
        try:
            if not responses:
                return {}
            
            # 按分数排序
            sorted_responses = sorted(
                responses, 
                key=lambda x: x.get('total_score', 0), 
                reverse=True
            )
            
            # 计算统计信息
            scores = [r.get('total_score', 0) for r in responses]
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 找出最佳和最差响应
            best_response = sorted_responses[0]
            worst_response = sorted_responses[-1]
            
            # 生成对比报告
            comparison = {
                'total_responses': len(responses),
                'score_statistics': {
                    'average': round(avg_score, 2),
                    'standard_deviation': round(std_score, 2),
                    'highest': round(max(scores), 2),
                    'lowest': round(min(scores), 2)
                },
                'ranking': [
                    {
                        'rank': i + 1,
                        'provider': r.get('provider', 'Unknown'),
                        'score': r.get('total_score', 0),
                        'strongest_dimension': r.get('detailed_report', {}).get('strongest_dimension', {}).get('name', 'Unknown')
                    }
                    for i, r in enumerate(sorted_responses)
                ],
                'best_response': {
                    'provider': best_response.get('provider', 'Unknown'),
                    'score': best_response.get('total_score', 0),
                    'strengths': best_response.get('detailed_report', {}).get('strongest_dimension', {})
                },
                'worst_response': {
                    'provider': worst_response.get('provider', 'Unknown'),
                    'score': worst_response.get('total_score', 0),
                    'weaknesses': worst_response.get('detailed_report', {}).get('weakest_dimension', {})
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing responses: {e}")
            return {}