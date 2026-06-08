# backend/data/__init__.py
"""
数据处理模块
"""
from .collector import DataCollector
from .database import RecommendationDatabase

__all__ = [
    'DataCollector',
    'RecommendationDatabase'
]

