# backend/data/collector.py
"""
数据收集模块 - 收集用户交互和反馈数据
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, data_dir: str = "./data/collected"):
        """
        初始化数据收集器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 数据缓冲区（用于批量写入）
        self.buffer = []
        self.buffer_size = 100  # 缓冲区大小
        
        logger.info(f"🚀 Data collector initialized (data_dir: {data_dir})")
    
    def collect_interaction(
        self,
        query: str,
        user_id: str,
        provider: str,
        response: str,
        recommendation_result: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        收集用户交互数据
        
        Args:
            query: 用户查询
            user_id: 用户ID
            provider: 使用的AI提供商
            response: AI响应
            recommendation_result: 推荐结果（可选）
            metadata: 额外元数据（可选）
            
        Returns:
            记录ID
        """
        try:
            record_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id[:8]}_{provider}"
            
            interaction = {
                'id': record_id,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'query': query,
                'provider': provider,
                'response': response[:1000],  # 限制长度
                'query_length': len(query),
                'response_length': len(response),
                'recommendation': recommendation_result,
                'metadata': metadata or {}
            }
            
            # 添加到缓冲区
            self.buffer.append(interaction)
            
            # 如果缓冲区满了，写入文件
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
            
            logger.debug(f"📝 Collected interaction: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"❌ Error collecting interaction: {e}")
            return ""
    
    def collect_feedback(
        self,
        interaction_id: str,
        user_id: str,
        provider: str,
        satisfaction: float,
        quality: Optional[float] = None,
        helpful: Optional[bool] = None,
        accurate: Optional[bool] = None,
        fast: Optional[bool] = None,
        comments: Optional[str] = None
    ):
        """
        收集用户反馈
        
        Args:
            interaction_id: 交互记录ID
            user_id: 用户ID
            provider: AI提供商
            satisfaction: 满意度（0-1）
            quality: 质量评分（0-1，可选）
            helpful: 是否有帮助（可选）
            accurate: 是否准确（可选）
            fast: 是否快速（可选）
            comments: 用户评论（可选）
        """
        try:
            feedback = {
                'interaction_id': interaction_id,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'provider': provider,
                'satisfaction': max(0.0, min(1.0, satisfaction)),
                'quality': quality if quality is not None else satisfaction,
                'helpful': helpful,
                'accurate': accurate,
                'fast': fast,
                'comments': comments
            }
            
            # 保存反馈到单独文件
            feedback_file = os.path.join(self.data_dir, f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl")
            
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + '\n')
            
            logger.info(f"✅ Collected feedback: satisfaction={satisfaction:.2f}, provider={provider}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting feedback: {e}")
    
    def collect_recommendation_result(
        self,
        query: str,
        user_id: str,
        recommended_provider: str,
        actual_provider: Optional[str] = None,
        user_accepted: bool = True,
        recommendation_score: Optional[float] = None
    ):
        """
        收集推荐结果数据
        
        Args:
            query: 用户查询
            user_id: 用户ID
            recommended_provider: 推荐的提供商
            actual_provider: 实际使用的提供商（如果用户选择了不同的）
            user_accepted: 用户是否接受了推荐
            recommendation_score: 推荐分数
        """
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'query': query[:500],  # 限制长度
                'recommended_provider': recommended_provider,
                'actual_provider': actual_provider or recommended_provider,
                'user_accepted': user_accepted,
                'recommendation_score': recommendation_score,
                'recommendation_accuracy': user_accepted and (actual_provider == recommended_provider or actual_provider is None)
            }
            
            # 保存到推荐结果文件
            rec_file = os.path.join(self.data_dir, f"recommendations_{datetime.now().strftime('%Y%m%d')}.jsonl")
            
            with open(rec_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.debug(f"📊 Collected recommendation result: {recommended_provider}, accepted={user_accepted}")
            
        except Exception as e:
            logger.error(f"❌ Error collecting recommendation result: {e}")
    
    def _flush_buffer(self):
        """将缓冲区数据写入文件"""
        if not self.buffer:
            return
        
        try:
            # 按日期分组写入
            today = datetime.now().strftime('%Y%m%d')
            interaction_file = os.path.join(self.data_dir, f"interactions_{today}.jsonl")
            
            with open(interaction_file, 'a', encoding='utf-8') as f:
                for record in self.buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"💾 Flushed {len(self.buffer)} interactions to file")
            self.buffer = []
            
        except Exception as e:
            logger.error(f"❌ Error flushing buffer: {e}")
    
    def flush(self):
        """手动刷新缓冲区"""
        self._flush_buffer()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取收集统计信息"""
        try:
            # 统计今天的交互数
            today = datetime.now().strftime('%Y%m%d')
            interaction_file = os.path.join(self.data_dir, f"interactions_{today}.jsonl")
            
            interaction_count = 0
            if os.path.exists(interaction_file):
                with open(interaction_file, 'r', encoding='utf-8') as f:
                    interaction_count = sum(1 for _ in f)
            
            # 统计今天的反馈数
            feedback_file = os.path.join(self.data_dir, f"feedback_{today}.jsonl")
            feedback_count = 0
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_count = sum(1 for _ in f)
            
            return {
                'buffer_size': len(self.buffer),
                'interactions_today': interaction_count,
                'feedback_today': feedback_count,
                'data_dir': self.data_dir
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {}

