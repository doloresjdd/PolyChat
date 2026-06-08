# 智能AI推荐系统文档

## 概述

本系统是一个基于机器学习的智能AI推荐系统，能够帮助用户从4个AI模型（OpenAI、Claude、Gemini、Ollama）中选择最合适的一个。系统通过持续学习用户偏好和AI能力差异来不断优化推荐效果。

## 系统架构

### 核心模块

1. **特征提取器 (feature_extractor.py)**
   - 从用户查询中提取特征（查询类型、复杂度、语言等）
   - 从用户历史中提取偏好特征
   - 从AI提供商历史中提取性能特征

2. **推荐模型 (recommender.py)**
   - 基于多维度特征计算推荐分数
   - 综合考虑查询类型匹配、用户偏好、提供商性能、响应时间、成本效率
   - 生成推荐理由和置信度

3. **Thompson Sampling (bandit.py)**
   - 多臂老虎机算法实现探索-利用平衡
   - 在利用已知最佳提供商和探索新提供商之间找到平衡
   - 持续更新成功/失败统计

4. **数据收集 (collector.py)**
   - 收集用户交互数据
   - 收集用户反馈数据
   - 收集推荐结果数据

5. **数据库 (database.py)**
   - SQLite数据库存储所有交互、反馈和推荐记录
   - 支持查询用户历史、提供商性能等

6. **模型训练 (trainer.py)**
   - 基于反馈数据持续训练模型
   - 评估推荐准确性
   - 优化特征权重

## API接口

### 1. 获取推荐

**端点**: `POST /api/recommend/`

**请求体**:
```json
{
  "query": "如何优化Python代码性能？",
  "user_id": "user123",
  "available_providers": ["openai", "claude", "gemini", "ollama"]
}
```

**响应**:
```json
{
  "status": "success",
  "recommended_provider": "claude",
  "recommendation_score": 0.85,
  "recommendation_reason": "Claude performs best on technical queries | User has high satisfaction (0.82) with Claude",
  "confidence": 0.78,
  "query_type": "technical",
  "all_scores": {
    "openai": 0.72,
    "claude": 0.85,
    "gemini": 0.68,
    "ollama": 0.55
  },
  "alternative_providers": [
    {"provider": "openai", "score": 0.72, "score_difference": 0.13}
  ]
}
```

### 2. 提交反馈

**端点**: `POST /api/feedback/`

**请求体**:
```json
{
  "interaction_id": "interaction123",
  "user_id": "user123",
  "provider": "claude",
  "satisfaction": 0.9,
  "quality": 0.85,
  "helpful": true,
  "accurate": true,
  "fast": true,
  "comments": "非常准确的回答"
}
```

**响应**:
```json
{
  "status": "success",
  "message": "Feedback submitted successfully",
  "timestamp": "2024-01-01T12:00:00"
}
```

### 3. 智能推荐聊天（集成端点）

**端点**: `POST /api/chat/smart-recommend`

这是集成到现有聊天系统的端点，会自动：
1. 获取推荐
2. 调用推荐的AI提供商
3. 返回响应和推荐信息

## 推荐算法详解

### 特征权重

- **查询类型匹配度** (30%): 提供商在特定查询类型上的历史表现
- **用户偏好** (25%): 用户历史使用频率和满意度
- **提供商性能** (25%): 提供商的历史成功率和质量
- **响应时间** (10%): 平均响应速度
- **成本效率** (10%): API调用成本

### Thompson Sampling

系统使用Thompson Sampling算法在以下场景中平衡探索和利用：

- **探索**: 尝试使用较少使用的提供商，收集更多数据
- **利用**: 优先使用历史表现最好的提供商

算法会根据Beta分布采样，结合推荐分数来选择提供商。

## 数据流程

1. **用户发送查询** → 系统提取特征
2. **获取推荐** → 基于特征和历史数据计算推荐分数
3. **调用AI** → 使用推荐的提供商生成响应
4. **用户反馈** → 收集满意度、质量等反馈
5. **模型更新** → 根据反馈更新bandit统计和模型参数

## 配置

配置文件位于 `backend/config/settings.py`，可以调整：

- 特征权重
- 提供商成本
- 数据收集缓冲区大小
- 训练间隔等

## 使用示例

### Python代码示例

```python
from ml.recommender import AIRecommender
from data.database import RecommendationDatabase

# 初始化
recommender = AIRecommender()
database = RecommendationDatabase()

# 获取推荐
recommendation = recommender.recommend(
    query="如何实现一个推荐系统？",
    user_id="user123",
    user_history=user_history,
    provider_history=provider_history
)

print(f"推荐: {recommendation['recommended_provider']}")
print(f"理由: {recommendation['recommendation_reason']}")

# 提交反馈
recommender.update_from_feedback(
    provider="claude",
    query="如何实现一个推荐系统？",
    user_id="user123",
    feedback={"satisfaction": 0.9, "quality": 0.85}
)
```

## 持续学习

系统支持持续学习：

1. **实时更新**: 每次反馈都会更新bandit统计
2. **定期训练**: 可以设置定期训练任务来优化模型
3. **性能评估**: 定期评估推荐准确性并生成报告

## 未来改进方向

1. **深度学习模型**: 使用神经网络学习更复杂的特征组合
2. **上下文感知**: 考虑对话历史和上下文
3. **多目标优化**: 同时优化质量、速度、成本等多个目标
4. **A/B测试**: 支持不同推荐策略的A/B测试
5. **冷启动问题**: 为新用户和新的查询类型提供更好的推荐

## 注意事项

1. 首次使用时，系统需要收集足够的数据才能做出准确推荐
2. 建议定期查看推荐统计和准确性报告
3. 可以根据实际使用情况调整特征权重
4. 数据库文件会持续增长，建议定期清理旧数据

