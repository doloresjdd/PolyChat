# backend/ml_services/response_scorer.py
"""
响应评分系统 - 为每个AI的响应打分
"""
import re
from typing import Dict, List
from textblob import TextBlob
import spacy
from collections import Counter

class ResponseScorer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # 如果没有安装，先下载
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def score_response(self, query: str, response: str) -> Dict[str, float]:
        """
        多维度评分系统
        """
        scores = {}
        
        # 1. 相关性评分
        scores['relevance'] = self._calculate_relevance(query, response)
        
        # 2. 完整性评分
        scores['completeness'] = self._calculate_completeness(response)
        
        # 3. 清晰度评分
        scores['clarity'] = self._calculate_clarity(response)
        
        # 4. 技术性评分（检测代码、专业术语等）
        scores['technical'] = self._calculate_technical_score(response)
        
        # 5. 情感积极度
        scores['sentiment'] = self._calculate_sentiment(response)
        
        # 总体评分
        scores['overall'] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """计算相关性"""
        query_doc = self.nlp(query.lower())
        response_doc = self.nlp(response.lower())
        
        # 提取关键词
        query_keywords = {token.lemma_ for token in query_doc 
                         if not token.is_stop and token.is_alpha}
        response_keywords = {token.lemma_ for token in response_doc 
                           if not token.is_stop and token.is_alpha}
        
        if not query_keywords:
            return 0.5
        
        # 计算重叠度
        overlap = len(query_keywords & response_keywords)
        relevance = min(overlap / len(query_keywords), 1.0)
        
        # 如果回答直接引用了问题，加分
        if any(keyword in response.lower() for keyword in query.lower().split()):
            relevance = min(relevance + 0.2, 1.0)
        
        return relevance
    
    def _calculate_completeness(self, response: str) -> float:
        """计算完整性"""
        word_count = len(response.split())
        
        # 基于长度的评分
        if word_count < 20:
            return 0.2
        elif word_count < 50:
            return 0.4
        elif word_count < 150:
            return 0.7
        elif word_count < 500:
            return 0.9
        else:
            return 0.85  # 太长也要扣分
    
    def _calculate_clarity(self, response: str) -> float:
        """计算清晰度"""
        sentences = response.split('.')
        
        # 平均句子长度
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # 理想句子长度是15-20个词
        if 15 <= avg_sentence_length <= 20:
            clarity_score = 1.0
        elif avg_sentence_length < 10:
            clarity_score = 0.6
        elif avg_sentence_length > 30:
            clarity_score = 0.5
        else:
            clarity_score = 0.8
        
        # 检查是否有结构（段落、列表等）
        has_structure = bool(re.search(r'(\n\n|^\d+\.|^-\s|^\*\s)', response, re.MULTILINE))
        if has_structure:
            clarity_score = min(clarity_score + 0.1, 1.0)
        
        return clarity_score
    
    def _calculate_technical_score(self, response: str) -> float:
        """计算技术性分数"""
        score = 0.5  # 基础分
        
        # 检查代码块
        if '```' in response or re.search(r'(def |class |function |import |const |let |var )', response):
            score += 0.3
        
        # 检查技术术语
        tech_terms = ['algorithm', 'function', 'variable', 'database', 'api', 
                     'framework', 'library', 'method', 'parameter', 'array']
        tech_count = sum(1 for term in tech_terms if term in response.lower())
        score += min(tech_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_sentiment(self, response: str) -> float:
        """计算情感积极度"""
        try:
            blob = TextBlob(response)
            # polarity范围是-1到1，转换为0到1
            sentiment = (blob.sentiment.polarity + 1) / 2
            return sentiment
        except:
            return 0.5  # 默认中性

    def compare_responses(self, query: str, responses: Dict[str, str]) -> Dict[str, Dict]:
        """
        比较多个AI的响应
        """
        results = {}
        
        for ai_name, response in responses.items():
            scores = self.score_response(query, response)
            results[ai_name] = {
                'response': response,
                'scores': scores,
                'rank': 0  # 稍后计算排名
            }
        
        # 计算排名
        sorted_ais = sorted(results.keys(), 
                          key=lambda x: results[x]['scores']['overall'], 
                          reverse=True)
        
        for rank, ai_name in enumerate(sorted_ais, 1):
            results[ai_name]['rank'] = rank
        
        return results