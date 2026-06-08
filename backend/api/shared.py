# backend/api/shared.py
"""
共享单例 - 确保推荐模型和反馈使用同一个实例
"""
from ml.recommender import AIRecommender
from data.database import RecommendationDatabase
from data.collector import DataCollector

# 全局单例
recommender = AIRecommender()
database = RecommendationDatabase()
collector = DataCollector()
