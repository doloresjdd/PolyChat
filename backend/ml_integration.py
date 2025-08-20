# backend/ml_integration.py
"""
ML增强功能集成模块
"""
from typing import Dict, List, Optional, Tuple
import asyncio
import time
from datetime import datetime
import json

from ml_services.response_scorer import ResponseScorer
from ml_services.simple_cache_manager import SmartCacheManager
from ml_services.api_optimizer import APIOptimizer

# 导入现有的providers
from llm_providers import openai_provider, ollama_provider, claude_provider, gemini_provider

class MLEnhancedPolyChat:
    def __init__(self):
        """初始化ML组件"""
        print("Initializing ML components...")
        self.scorer = ResponseScorer()
        self.cache = SmartCacheManager()
        self.optimizer = APIOptimizer()
        
        # Provider映射
        self.providers = {
            'openai': openai_provider,
            'ollama': ollama_provider,
            'claude': claude_provider,
            'gemini': gemini_provider
        }
        
        # 性能记录
        self.performance_history = []
        
    async def process_enhanced_query(
        self, 
        prompt: str, 
        history: List = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
        providers_to_use: Optional[List[str]] = None
    ) -> Dict:
        """
        增强版查询处理 - 包含缓存、评分、优化
        """
        start_time = time.time()
        
        # 1. 检查缓存
        if use_cache:
            cached_result = self.cache.search_similar(prompt)
            if cached_result and cached_result['similarity'] > 0.95:
                print(f"Cache hit! Similarity: {cached_result['similarity']:.2f}")
                return {
                    'prompt': prompt,
                    'responses': cached_result['responses'],
                    'scores': cached_result['scores'],
                    'cache_hit': True,
                    'similarity': cached_result['similarity'],
                    'process_time': time.time() - start_time,
                    'recommended_provider': self._get_best_provider(cached_result['scores'])
                }
        
        # 2. 分类查询
        query_type = self.optimizer.classify_query(prompt)
        
        # 3. 决定使用哪些providers
        if providers_to_use is None:
            providers_to_use = self._select_providers(query_type)
        
        # 4. 并发调用多个providers
        responses = await self._call_multiple_providers(prompt, history, providers_to_use)
        
        # 5. 评分所有响应
        comparison_results = self.scorer.compare_responses(prompt, responses)
        
        # 6. 添加到缓存
        if use_cache:
            self.cache.add_to_cache(prompt, responses, comparison_results)
        
        # 7. 更新优化器性能数据
        for provider_name, result in comparison_results.items():
            success = result['scores']['overall'] > 0.6
            response_time = result.get('response_time', 1.0)
            self.optimizer.update_performance(provider_name, success, response_time)
        
        # 8. 记录性能历史
        self._record_performance(prompt, query_type, comparison_results)
        
        # 9. 获取最佳provider
        best_provider = self._get_best_provider(comparison_results)
        
        return {
            'prompt': prompt,
            'query_type': query_type,
            'responses': comparison_results,
            'recommended_provider': best_provider,
            'cache_hit': False,
            'process_time': time.time() - start_time,
            'providers_used': list(responses.keys())
        }
    
    async def process_single_provider(
        self,
        provider: str,
        prompt: str,
        history: List = None,
        evaluate: bool = True
    ) -> Dict:
        """
        使用单个provider处理查询（向后兼容）
        """
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        start_time = time.time()
        
        # 调用provider
        response = await self.providers[provider].generate_response(prompt, history or [])
        
        result = {
            'provider': provider,
            'response': response,
            'response_time': time.time() - start_time
        }
        
        # 可选：评分
        if evaluate:
            scores = self.scorer.score_response(prompt, response)
            result['scores'] = scores
        
        return result
    
    async def _call_multiple_providers(
        self,
        prompt: str,
        history: List,
        providers_to_use: List[str]
    ) -> Dict[str, str]:
        """
        并发调用多个providers
        """
        tasks = []
        provider_names = []
        
        for provider_name in providers_to_use:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                task = provider.generate_response(prompt, history or [])
                tasks.append(task)
                provider_names.append(provider_name)
        
        # 并发执行所有任务
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 组合结果
        result = {}
        for provider_name, response in zip(provider_names, responses):
            if isinstance(response, Exception):
                print(f"Error from {provider_name}: {response}")
                result[provider_name] = f"Error: {str(response)}"
            else:
                result[provider_name] = response
        
        return result
    
    def _select_providers(self, query_type: str) -> List[str]:
        """
        根据查询类型选择要使用的providers
        """
        # 基于查询类型的策略
        if query_type == 'code':
            # 编程问题：OpenAI和Claude最好
            return ['openai', 'claude']
        elif query_type == 'simple':
            # 简单问题：用快速便宜的
            return ['gemini', 'ollama']
        elif query_type == 'creative':
            # 创意写作：Claude和OpenAI
            return ['claude', 'openai']
        else:
            # 一般问题：用Thompson Sampling选择2-3个
            all_providers = list(self.providers.keys())
            selected = []
            
            # 选择2-3个providers
            for _ in range(min(3, len(all_providers))):
                if all_providers:
                    provider = self.optimizer.select_api(query_type, all_providers)
                    selected.append(provider)
                    all_providers.remove(provider)
            
            return selected
    
    def _get_best_provider(self, comparison_results: Dict) -> str:
        """
        获取得分最高的provider
        """
        best_provider = None
        best_score = -1
        
        for provider_name, result in comparison_results.items():
            if isinstance(result, dict) and 'scores' in result:
                overall_score = result['scores'].get('overall', 0)
                if overall_score > best_score:
                    best_score = overall_score
                    best_provider = provider_name
        
        return best_provider or 'openai'  # 默认返回openai
    
    def _record_performance(self, prompt: str, query_type: str, results: Dict):
        """
        记录性能数据用于分析
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt[:100],  # 只记录前100字符
            'query_type': query_type,
            'provider_scores': {}
        }
        
        for provider, result in results.items():
            if isinstance(result, dict) and 'scores' in result:
                record['provider_scores'][provider] = result['scores']['overall']
        
        self.performance_history.append(record)
        
        # 只保留最近1000条记录
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        """
        return {
            'cache_stats': self.cache.get_cache_stats(),
            'api_stats': self.optimizer.get_api_stats(),
            'total_queries': len(self.performance_history),
            'recent_performance': self.performance_history[-10:] if self.performance_history else []
        }

# 创建全局实例
ml_system = MLEnhancedPolyChat()