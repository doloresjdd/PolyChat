# backend/ml/bandit.py
"""
Thompson Sampling - 多臂老虎机算法用于探索-利用平衡
"""
import numpy as np
import logging
from typing import Dict, List, Optional
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)

class ThompsonSamplingBandit:
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化Thompson Sampling算法
        
        Args:
            config_file: 配置文件路径（用于持久化）
        """
        self.config_file = config_file or "./models/bandit_config.json"

        # 冷启动先验：基于业界 benchmark 的领域知识
        # alpha = 成功次数+1, beta = 失败次数+1
        # 数值越大代表先验越强（等效于已观测到的样本数）
        self.COLD_START_PRIORS = {
            # (alpha, beta) 对应 Beta 分布均值 = alpha/(alpha+beta)
            'code':         {'openai': (12, 3), 'claude': (10, 4), 'gemini': (7, 5),  'ollama': (4, 8)},
            'creative':     {'openai': (8,  4), 'claude': (12, 2), 'gemini': (6, 6),  'ollama': (3, 9)},
            'analytical':   {'openai': (10, 4), 'claude': (11, 3), 'gemini': (10, 3), 'ollama': (4, 8)},
            'math':         {'openai': (12, 2), 'claude': (9,  4), 'gemini': (11, 3), 'ollama': (3, 9)},
            'simple':       {'openai': (8,  4), 'claude': (8,  4), 'gemini': (9, 3),  'ollama': (7, 5)},
            'technical':    {'openai': (10, 3), 'claude': (10, 3), 'gemini': (8, 4),  'ollama': (5, 7)},
            'conversational':{'openai':(8,  4), 'claude': (9,  3), 'gemini': (8, 4),  'ollama': (6, 6)},
            'privacy':      {'openai': (3, 10), 'claude': (3, 10), 'gemini': (2, 11), 'ollama': (14, 1)},
            'general':      {'openai': (9,  4), 'claude': (9,  4), 'gemini': (8, 4),  'ollama': (5, 7)},
        }

        # Beta分布参数（成功和失败次数）
        # 若有持久化数据则加载，否则使用无信息先验（具体 query_type 的先验在 select_arm 时按需注入）
        self.alpha = defaultdict(lambda: 1.0)  # 成功次数 + 1
        self.beta = defaultdict(lambda: 1.0)   # 失败次数 + 1

        # 加载历史数据
        self.load_config()
        
        logger.info("🚀 Thompson Sampling Bandit initialized")
    
    def select_arm(
        self,
        scores: Dict[str, float],
        available_arms: Optional[List[str]] = None,
        query_type: Optional[str] = None
    ) -> str:
        """
        使用Thompson Sampling选择臂（AI提供商）
        
        Args:
            scores: 各提供商的推荐分数
            available_arms: 可用的臂列表（如果为None，使用scores的keys）
            
        Returns:
            选择的提供商名称
        """
        if available_arms is None:
            available_arms = list(scores.keys())
        
        if not available_arms:
            return 'openai'  # 默认
        
        # 如果只有一个选项，直接返回
        if len(available_arms) == 1:
            return available_arms[0]
        
        # 决定使用冷启动先验还是已学习的参数
        # 若某个 arm 的观测总数 < 5，认为数据不足，混入先验
        priors = self.COLD_START_PRIORS.get(query_type or 'general', self.COLD_START_PRIORS['general'])

        # 从Beta分布采样
        samples = {}
        for arm in available_arms:
            observed_total = self.alpha[arm] + self.beta[arm] - 2  # 减去初始化的1+1
            if observed_total < 5 and arm in priors:
                # 冷启动：直接使用先验参数
                alpha = priors[arm][0]
                beta  = priors[arm][1]
            else:
                # 已有足够数据：使用学到的参数
                alpha = self.alpha[arm]
                beta  = self.beta[arm]
            
            # 从Beta分布采样
            sample = np.random.beta(alpha, beta)
            
            # 结合推荐分数（加权平均）
            # 70%来自历史表现，30%来自当前推荐分数
            combined_score = 0.7 * sample + 0.3 * scores.get(arm, 0.5)
            
            samples[arm] = combined_score
        
        # 选择最高采样值的臂
        selected_arm = max(samples.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"🎰 Selected arm: {selected_arm} (samples: {samples})")
        return selected_arm
    
    def update(self, arm: str, success: bool, reward: Optional[float] = None):
        """
        更新臂的统计信息
        
        Args:
            arm: 选择的臂（提供商）
            success: 是否成功
            reward: 奖励值（可选，0-1之间）
        """
        if success:
            self.alpha[arm] += 1.0
            if reward is not None:
                # 如果提供了奖励值，根据奖励调整
                self.alpha[arm] += reward
        else:
            self.beta[arm] += 1.0
            if reward is not None:
                # 失败时，奖励值越小，beta增加越多
                self.beta[arm] += (1.0 - reward)
        
        # 保存配置
        self.save_config()
        
        logger.debug(f"📊 Updated {arm}: alpha={self.alpha[arm]:.2f}, beta={self.beta[arm]:.2f}")
    
    def get_expected_reward(self, arm: str) -> float:
        """
        获取臂的期望奖励
        
        Args:
            arm: 臂名称
            
        Returns:
            期望奖励值（0-1之间）
        """
        alpha = self.alpha[arm]
        beta = self.beta[arm]
        
        # Beta分布的期望值 = alpha / (alpha + beta)
        if alpha + beta == 0:
            return 0.5
        
        return alpha / (alpha + beta)
    
    def get_confidence_interval(self, arm: str, confidence: float = 0.95) -> tuple:
        """
        获取臂的置信区间
        
        Args:
            arm: 臂名称
            confidence: 置信水平（默认0.95）
            
        Returns:
            (下界, 上界)
        """
        alpha = self.alpha[arm]
        beta = self.beta[arm]
        
        if alpha + beta <= 2:
            return (0.0, 1.0)
        
        # 使用Beta分布的置信区间
        try:
            from scipy import stats
            lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
            upper = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
        except ImportError:
            # 如果scipy不可用，使用近似值
            mean = alpha / (alpha + beta)
            std = np.sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            z_score = 1.96  # 95%置信区间
            lower = max(0.0, mean - z_score * std)
            upper = min(1.0, mean + z_score * std)
        
        return (lower, upper)
    
    def get_statistics(self) -> Dict[str, Dict]:
        """
        获取所有臂的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        for arm in set(list(self.alpha.keys()) + list(self.beta.keys())):
            alpha = self.alpha[arm]
            beta = self.beta[arm]
            total = alpha + beta - 2  # 减去初始的1+1
            
            stats[arm] = {
                'successes': max(0, alpha - 1),
                'failures': max(0, beta - 1),
                'total_trials': max(0, total),
                'success_rate': self.get_expected_reward(arm),
                'confidence_interval': self.get_confidence_interval(arm) if total > 0 else (0.0, 1.0)
            }
        
        return stats
    
    def reset_arm(self, arm: str):
        """重置臂的统计信息"""
        self.alpha[arm] = 1.0
        self.beta[arm] = 1.0
        self.save_config()
        logger.info(f"🔄 Reset statistics for {arm}")
    
    def reset_all(self):
        """重置所有统计信息"""
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        self.save_config()
        logger.info("🔄 Reset all statistics")
    
    def load_config(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.alpha = defaultdict(lambda: 1.0, config.get('alpha', {}))
                    self.beta = defaultdict(lambda: 1.0, config.get('beta', {}))
                logger.info("✅ Loaded bandit configuration")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load config: {e}")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config = {
                'alpha': dict(self.alpha),
                'beta': dict(self.beta)
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save config: {e}")

