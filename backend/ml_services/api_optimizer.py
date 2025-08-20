# backend/ml_services/api_optimizer.py
"""
API优化器 - 使用强化学习选择最佳API
"""
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import json
import os

class APIOptimizer:
    def __init__(self, config_file: str = "./api_config.json"):
        self.config_file = config_file
        self.load_config()
        
        # Thompson Sampling参数
        self.successes = defaultdict(lambda: 1)  # 成功次数
        self.failures = defaultdict(lambda: 1)   # 失败次数
        
        # 响应时间记录
        self.response_times = defaultdict(list)
        
        # API成本（每1000 tokens）
        self.costs = {
            'openai': 0.002,
            'claude': 0.008,
            'gemini': 0.001,
            'ollama': 0.0  # 本地免费
        }
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.successes = defaultdict(lambda: 1, config.get('successes', {}))
                self.failures = defaultdict(lambda: 1, config.get('failures', {}))
        
    def save_config(self):
        """保存配置"""
        config = {
            'successes': dict(self.successes),
            'failures': dict(self.failures)
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
    
    def select_api(self, query_type: str = 'general', 
                   available_apis: List[str] = None,
                   budget_constraint: float = None) -> str:
        """
        使用Thompson Sampling选择API
        """
        if available_apis is None:
            available_apis = list(self.costs.keys())
        
        # 根据预算过滤
        if budget_constraint:
            available_apis = [api for api in available_apis 
                            if self.costs.get(api, 0) <= budget_constraint]
        
        # 特殊查询类型的规则
        if query_type == 'code':
            # 编程问题优先选择GPT-4或Claude
            preferred = ['openai', 'claude']
            available_preferred = [api for api in preferred if api in available_apis]
            if available_preferred:
                return self._thompson_sampling(available_preferred)
        
        elif query_type == 'simple':
            # 简单问题选择便宜的
            return 'ollama' if 'ollama' in available_apis else 'gemini'
        
        # 一般情况使用Thompson Sampling
        return self._thompson_sampling(available_apis)
    
    def _thompson_sampling(self, apis: List[str]) -> str:
        """
        Thompson Sampling算法
        """
        samples = {}
        
        for api in apis:
            # 从Beta分布采样
            alpha = self.successes[api]
            beta = self.failures[api]
            samples[api] = np.random.beta(alpha, beta)
        
        # 选择最高采样值的API
        return max(samples, key=samples.get)
    
    def update_performance(self, api: str, success: bool, response_time: float = None):
        """
        更新API性能数据
        """
        if success:
            self.successes[api] += 1
        else:
            self.failures[api] += 1
        
        if response_time:
            if api not in self.response_times:
                self.response_times[api] = []
            self.response_times[api].append(response_time)
            
            # 只保留最近100次
            if len(self.response_times[api]) > 100:
                self.response_times[api] = self.response_times[api][-100:]
        
        self.save_config()
    
    def get_api_stats(self) -> Dict:
        """
        获取API统计信息
        """
        stats = {}
        
        for api in self.costs.keys():
            total_calls = self.successes[api] + self.failures[api] - 2
            if total_calls > 0:
                success_rate = (self.successes[api] - 1) / total_calls
                avg_time = np.mean(self.response_times[api]) if self.response_times[api] else 0
                
                stats[api] = {
                    'success_rate': success_rate,
                    'total_calls': total_calls,
                    'avg_response_time': avg_time,
                    'cost_per_call': self.costs[api]
                }
        
        return stats
    
    def classify_query(self, query: str) -> str:
        """
        简单的查询分类
        """
        query_lower = query.lower()
        
        # 编程相关
        code_keywords = ['code', 'function', 'debug', 'error', 'python', 'javascript', 
                        'program', 'algorithm', 'implement']
        if any(keyword in query_lower for keyword in code_keywords):
            return 'code'
        
        # 简单查询
        simple_keywords = ['what is', 'define', 'meaning', 'who is', 'when', 'where']
        if any(keyword in query_lower for keyword in simple_keywords) and len(query) < 50:
            return 'simple'
        
        # 创意写作
        creative_keywords = ['write', 'story', 'poem', 'creative', 'imagine', 'describe']
        if any(keyword in query_lower for keyword in creative_keywords):
            return 'creative'
        
        # 数据分析
        data_keywords = ['analyze', 'data', 'statistics', 'calculate', 'graph', 'chart']
        if any(keyword in query_lower for keyword in data_keywords):
            return 'analytical'
        
        return 'general'