# backend/data/database.py
"""
数据库操作模块 - 存储和查询用户交互、反馈和推荐数据
"""
import logging
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class RecommendationDatabase:
    def __init__(self, db_path: str = "./data/recommendation.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"🚀 Database initialized (path: {db_path})")
    
    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户交互表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                provider TEXT NOT NULL,
                response TEXT,
                query_type TEXT,
                recommendation_id TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 用户反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                satisfaction REAL NOT NULL,
                quality REAL,
                helpful INTEGER,
                accurate INTEGER,
                fast INTEGER,
                comments TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
        ''')
        
        # 推荐记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                recommended_provider TEXT NOT NULL,
                actual_provider TEXT,
                user_accepted INTEGER,
                recommendation_score REAL,
                recommendation_reason TEXT,
                query_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 提供商性能统计表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS provider_performance (
                provider TEXT NOT NULL,
                query_type TEXT,
                total_requests INTEGER DEFAULT 0,
                successful_requests INTEGER DEFAULT 0,
                total_response_time REAL DEFAULT 0,
                total_quality_score REAL DEFAULT 0,
                total_satisfaction REAL DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, query_type)
            )
        ''')
        
        # 用户偏好表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                avg_satisfaction REAL DEFAULT 0.5,
                last_used TEXT,
                PRIMARY KEY (user_id, provider)
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_provider ON interactions(provider)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_provider ON feedback(provider)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id)')
        
        conn.commit()
        conn.close()
    
    def save_interaction(
        self,
        interaction_id: str,
        user_id: str,
        query: str,
        provider: str,
        response: str,
        query_type: Optional[str] = None,
        recommendation_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """保存交互记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO interactions 
                (id, timestamp, user_id, query, provider, response, query_type, recommendation_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_id,
                datetime.now().isoformat(),
                user_id,
                query,
                provider,
                response[:5000],  # 限制长度
                query_type,
                recommendation_id,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            logger.debug(f"💾 Saved interaction: {interaction_id}")
            
        except Exception as e:
            logger.error(f"❌ Error saving interaction: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_feedback(
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
        """保存用户反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO feedback 
                (interaction_id, timestamp, user_id, provider, satisfaction, quality, 
                 helpful, accurate, fast, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_id,
                datetime.now().isoformat(),
                user_id,
                provider,
                satisfaction,
                quality,
                1 if helpful else 0 if helpful is False else None,
                1 if accurate else 0 if accurate is False else None,
                1 if fast else 0 if fast is False else None,
                comments
            ))
            
            # 更新用户偏好
            self._update_user_preference(user_id, provider, satisfaction)
            
            # 更新提供商性能
            interaction = self.get_interaction(interaction_id)
            if interaction:
                query_type = interaction.get('query_type')
                self._update_provider_performance(provider, query_type, satisfaction, quality)
            
            conn.commit()
            logger.info(f"💾 Saved feedback: satisfaction={satisfaction:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error saving feedback: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def save_recommendation(
        self,
        user_id: str,
        query: str,
        recommended_provider: str,
        actual_provider: Optional[str] = None,
        user_accepted: Optional[bool] = None,
        recommendation_score: Optional[float] = None,
        recommendation_reason: Optional[str] = None,
        query_type: Optional[str] = None
    ) -> int:
        """保存推荐记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO recommendations 
                (timestamp, user_id, query, recommended_provider, actual_provider, 
                 user_accepted, recommendation_score, recommendation_reason, query_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                user_id,
                query,
                recommended_provider,
                actual_provider,
                1 if user_accepted else 0 if user_accepted is False else None,
                recommendation_score,
                recommendation_reason,
                query_type
            ))
            
            recommendation_id = cursor.lastrowid
            conn.commit()
            logger.debug(f"💾 Saved recommendation: ID={recommendation_id}")
            return recommendation_id
            
        except Exception as e:
            logger.error(f"❌ Error saving recommendation: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def get_user_history(
        self,
        user_id: str,
        limit: int = 100,
        provider: Optional[str] = None
    ) -> List[Dict]:
        """获取用户历史交互"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if provider:
                cursor.execute('''
                    SELECT * FROM interactions 
                    WHERE user_id = ? AND provider = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, provider, limit))
            else:
                cursor.execute('''
                    SELECT * FROM interactions 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Error getting user history: {e}")
            return []
        finally:
            conn.close()
    
    def get_provider_history(
        self,
        provider: str,
        query_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """获取提供商历史记录"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if query_type:
                cursor.execute('''
                    SELECT * FROM interactions 
                    WHERE provider = ? AND query_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (provider, query_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM interactions 
                    WHERE provider = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (provider, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"❌ Error getting provider history: {e}")
            return []
        finally:
            conn.close()
    
    def get_interaction(self, interaction_id: str) -> Optional[Dict]:
        """获取单个交互记录"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM interactions WHERE id = ?', (interaction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"❌ Error getting interaction: {e}")
            return None
        finally:
            conn.close()
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Dict]:
        """获取用户偏好"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM user_preferences 
                WHERE user_id = ?
            ''', (user_id,))
            
            rows = cursor.fetchall()
            preferences = {}
            for row in rows:
                preferences[row['provider']] = {
                    'usage_count': row['usage_count'],
                    'avg_satisfaction': row['avg_satisfaction'],
                    'last_used': row['last_used']
                }
            
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Error getting user preferences: {e}")
            return {}
        finally:
            conn.close()
    
    def get_provider_performance(
        self,
        provider: str,
        query_type: Optional[str] = None
    ) -> Optional[Dict]:
        """获取提供商性能统计"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if query_type:
                cursor.execute('''
                    SELECT * FROM provider_performance 
                    WHERE provider = ? AND query_type = ?
                ''', (provider, query_type))
            else:
                # 获取所有类型的汇总
                cursor.execute('''
                    SELECT 
                        provider,
                        SUM(total_requests) as total_requests,
                        SUM(successful_requests) as successful_requests,
                        AVG(total_response_time / NULLIF(total_requests, 0)) as avg_response_time,
                        AVG(total_quality_score / NULLIF(total_requests, 0)) as avg_quality_score,
                        AVG(total_satisfaction / NULLIF(total_requests, 0)) as avg_satisfaction
                    FROM provider_performance
                    WHERE provider = ?
                    GROUP BY provider
                ''', (provider,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting provider performance: {e}")
            return None
        finally:
            conn.close()
    
    def _update_user_preference(self, user_id: str, provider: str, satisfaction: float):
        """更新用户偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 检查是否存在
            cursor.execute('''
                SELECT usage_count, avg_satisfaction FROM user_preferences
                WHERE user_id = ? AND provider = ?
            ''', (user_id, provider))
            
            row = cursor.fetchone()
            
            if row:
                # 更新
                usage_count = row[0] + 1
                old_avg = row[1]
                new_avg = (old_avg * (usage_count - 1) + satisfaction) / usage_count
                
                cursor.execute('''
                    UPDATE user_preferences
                    SET usage_count = ?, avg_satisfaction = ?, last_used = ?
                    WHERE user_id = ? AND provider = ?
                ''', (usage_count, new_avg, datetime.now().isoformat(), user_id, provider))
            else:
                # 插入
                cursor.execute('''
                    INSERT INTO user_preferences (user_id, provider, usage_count, avg_satisfaction, last_used)
                    VALUES (?, ?, 1, ?, ?)
                ''', (user_id, provider, satisfaction, datetime.now().isoformat()))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Error updating user preference: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _update_provider_performance(
        self,
        provider: str,
        query_type: Optional[str],
        satisfaction: float,
        quality: Optional[float]
    ):
        """更新提供商性能"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query_type = query_type or 'general'
        
        try:
            # 检查是否存在
            cursor.execute('''
                SELECT total_requests FROM provider_performance
                WHERE provider = ? AND query_type = ?
            ''', (provider, query_type))
            
            row = cursor.fetchone()
            
            if row:
                # 更新
                cursor.execute('''
                    UPDATE provider_performance
                    SET 
                        total_requests = total_requests + 1,
                        successful_requests = successful_requests + ?,
                        total_satisfaction = total_satisfaction + ?,
                        total_quality_score = total_quality_score + ?,
                        last_updated = ?
                    WHERE provider = ? AND query_type = ?
                ''', (
                    1 if satisfaction > 0.6 else 0,
                    satisfaction,
                    quality or satisfaction,
                    datetime.now().isoformat(),
                    provider,
                    query_type
                ))
            else:
                # 插入
                cursor.execute('''
                    INSERT INTO provider_performance 
                    (provider, query_type, total_requests, successful_requests, 
                     total_satisfaction, total_quality_score, last_updated)
                    VALUES (?, ?, 1, ?, ?, ?, ?)
                ''', (
                    provider,
                    query_type,
                    1 if satisfaction > 0.6 else 0,
                    satisfaction,
                    quality or satisfaction,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Error updating provider performance: {e}")
            conn.rollback()
        finally:
            conn.close()

