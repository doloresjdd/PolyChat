# backend/config/settings.py
"""
配置文件 - 推荐系统配置
"""
import os
from typing import Dict, Any

# 数据库配置
DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/recommendation.db")
DATA_COLLECTION_DIR = os.getenv("DATA_COLLECTION_DIR", "./data/collected")

# 模型配置
MODEL_PATH = os.getenv("MODEL_PATH", "./models/recommender_model.json")
BANDIT_CONFIG_PATH = os.getenv("BANDIT_CONFIG_PATH", "./models/bandit_config.json")

# 推荐系统配置
RECOMMENDATION_CONFIG: Dict[str, Any] = {
    'cache_threshold': 0.95,
    'max_providers_per_query': 3,
    'enable_learning': True,
    'enable_scoring': True,
    'enable_optimization': True,
    'feature_weights': {
        'query_type_match': 0.3,
        'user_preference': 0.25,
        'provider_performance': 0.25,
        'response_time': 0.1,
        'cost_efficiency': 0.1
    }
}

# AI提供商配置
PROVIDER_COSTS = {
    'openai': 0.002,   # 每1000 tokens
    'claude': 0.008,
    'gemini': 0.001,
    'ollama': 0.0      # 本地免费
}

# 数据收集配置
COLLECTOR_CONFIG = {
    'buffer_size': 100,
    'auto_flush_interval': 300  # 秒
}

# 训练配置
TRAINING_CONFIG = {
    'evaluation_days': 7,
    'retrain_interval': 24 * 60 * 60,  # 24小时
    'min_feedback_count': 100
}

