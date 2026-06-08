# backend/ml/__init__.py
"""
机器学习模块
"""
from .feature_extractor import FeatureExtractor
from .recommender import AIRecommender
from .bandit import ThompsonSamplingBandit
from .trainer import ModelTrainer

__all__ = [
    'FeatureExtractor',
    'AIRecommender',
    'ThompsonSamplingBandit',
    'ModelTrainer'
]

