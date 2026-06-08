# backend/ml_services/ml_integration.py
"""
ML增强功能集成模块 - 核心协调器
"""
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# 导入ML组件
from .response_scorer import ResponseScorer
from .simple_cache_manager import SmartCacheManager
from .api_optimizer import APIOptimizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLEnhancedPolyChat:
    def __init__(self, config: Optional[Dict] = None):
        """初始化ML增强系统"""
        logger.info("🚀 Initializing ML Enhanced PolyChat System...")
        
        # 加载配置
        self.config = config or self._load_default_config()
        
        # 初始化ML组件
        try:
            self.scorer = ResponseScorer()
            self.cache = SmartCacheManager(cache_dir="./cache")
            self.optimizer = APIOptimizer()
            logger.info("✅ All ML components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing ML components: {e}")
            raise
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'total_time_saved': 0.0,
            'api_calls_saved': 0
        }
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'cache_threshold': 0.95,
            'max_providers_per_query': 3,
            'enable_learning': True,
            'enable_scoring': True,
            'enable_optimization': True
        }
    
    async def process_query(
        self, 
        prompt: str, 
        user_id: Optional[str] = None,
        history: List[Dict] = None,
        use_cache: bool = True,
        providers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理用户查询 - 主要入口点
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        logger.info(f"📝 Processing query: {prompt[:50]}...")
        
        try:
            # 1. 缓存检查
            if use_cache:
                cached_result = await self._check_cache(prompt)
                if cached_result:
                    return await self._format_cached_response(cached_result, start_time)
            
            # 2. 查询分类和优化
            query_type = self.optimizer.classify_query(prompt)
            selected_providers = providers or self._select_providers(query_type)
            
            # 3. 并发调用AI提供商
            responses = await self._call_providers(prompt, selected_providers, history)
            
            # 4. 响应评分和比较
            scored_responses = await self._score_responses(prompt, responses)
            
            # 5. 更新缓存和学习
            if use_cache:
                await self._update_cache_and_learn(prompt, responses, scored_responses)
            
            # 6. 更新统计
            self._update_stats(start_time, use_cache=False)
            
            return await self._format_response(
                prompt, query_type, scored_responses, start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _check_cache(self, prompt: str) -> Optional[Dict]:
        """检查缓存"""
        try:
            cached = self.cache.search_similar(prompt, threshold=self.config['cache_threshold'])
            if cached:
                logger.info(f"�� Cache hit! Similarity: {cached['similarity']:.2f}")
                self.stats['cache_hits'] += 1
                return cached
        except Exception as e:
            logger.warning(f"⚠️ Cache check failed: {e}")
        return None
    
    async def _select_providers(self, query_type: str) -> List[str]:
        """选择要使用的AI提供商"""
        if self.config['enable_optimization']:
            # 使用智能优化器选择
            selected = []
            available_providers = ['openai', 'claude', 'gemini', 'ollama']
            
            for _ in range(min(self.config['max_providers_per_query'], len(available_providers))):
                if available_providers:
                    provider = self.optimizer.select_api(query_type, available_providers)
                    selected.append(provider)
                    available_providers.remove(provider)
            
            return selected
        else:
            # 使用默认选择
            return ['openai', 'claude']
    
    async def _call_providers(
        self, 
        prompt: str, 
        providers: List[str], 
        history: List[Dict]
    ) -> Dict[str, str]:
        """并发调用多个AI提供商"""
        logger.info(f"�� Calling providers: {providers}")
        
        # 这里需要集成你的实际AI提供商
        # 暂时返回模拟响应
        responses = {}
        for provider in providers:
            try:
                # 模拟API调用
                response = await self._mock_provider_call(provider, prompt)
                responses[provider] = response
            except Exception as e:
                logger.error(f"❌ Error calling {provider}: {e}")
                responses[provider] = f"Error: {str(e)}"
        
        return responses
    
    async def _mock_provider_call(self, provider: str, prompt: str) -> str:
        """模拟AI提供商调用（临时）"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return f"Mock response from {provider}: {prompt[:20]}..."
    
    async def _score_responses(self, prompt: str, responses: Dict[str, str]) -> Dict[str, Dict]:
        """评分所有响应"""
        if not self.config['enable_scoring']:
            return {provider: {'response': resp, 'scores': None} 
                   for provider, resp in responses.items()}
        
        scored = {}
        for provider, response in responses.items():
            try:
                scores = self.scorer.score_response(prompt, response)
                scored[provider] = {
                    'response': response,
                    'scores': scores,
                    'rank': 0
                }
            except Exception as e:
                logger.error(f"❌ Error scoring {provider} response: {e}")
                scored[provider] = {
                    'response': response,
                    'scores': {'overall': 0.0},
                    'rank': 999
                }
        
        # 计算排名
        sorted_providers = sorted(
            scored.keys(),
            key=lambda x: scored[x]['scores']['overall'] if scored[x]['scores'] else 0,
            reverse=True
        )
        
        for rank, provider in enumerate(sorted_providers, 1):
            scored[provider]['rank'] = rank
        
        return scored
    
    async def _update_cache_and_learn(self, prompt: str, responses: Dict, scored_responses: Dict):
        """更新缓存和学习"""
        try:
            # 添加到缓存
            self.cache.add_to_cache(prompt, responses, scored_responses)
            
            # 更新优化器性能数据
            if self.config['enable_learning']:
                for provider, result in scored_responses.items():
                    if result.get('scores'):
                        success = result['scores']['overall'] > 0.6
                        self.optimizer.update_performance(provider, success)
                        
        except Exception as e:
            logger.error(f"❌ Error updating cache/learning: {e}")
    
    async def _format_cached_response(self, cached_result: Dict, start_time: float) -> Dict:
        """格式化缓存响应"""
        self._update_stats(start_time, use_cache=True)
        
        return {
            'prompt': cached_result.get('cached_query', ''),
            'responses': cached_result['responses'],
            'scores': cached_result['scores'],
            'cache_hit': True,
            'similarity': cached_result['similarity'],
            'process_time': time.time() - start_time,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _format_response(
        self, 
        prompt: str, 
        query_type: str, 
        scored_responses: Dict, 
        start_time: float
    ) -> Dict:
        """格式化响应"""
        return {
            'prompt': prompt,
            'query_type': query_type,
            'responses': scored_responses,
            'cache_hit': False,
            'process_time': time.time() - start_time,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'recommended_provider': self._get_best_provider(scored_responses)
        }
    
    def _get_best_provider(self, scored_responses: Dict) -> str:
        """获取最佳提供商"""
        best_provider = None
        best_score = -1
        
        for provider, result in scored_responses.items():
            if result.get('scores') and result['scores']['overall'] > best_score:
                best_score = result['scores']['overall']
                best_provider = provider
        
        return best_provider or 'openai'
    
    def _update_stats(self, start_time: float, use_cache: bool):
        """更新统计信息"""
        if use_cache:
            time_saved = 2.0  # 假设缓存节省2秒
            self.stats['total_time_saved'] += time_saved
            self.stats['api_calls_saved'] += 1
    
    def get_system_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            'ml_system_stats': self.stats,
            'cache_stats': self.cache.get_cache_stats(),
            'optimizer_stats': self.optimizer.get_api_stats(),
            'config': self.config
        }
    
    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        try:
            return {
                'cache': self.cache is not None,
                'scorer': self.scorer is not None,
                'optimizer': self.optimizer is not None,
                'overall': True
            }
        except:
            return {'overall': False}

# 创建全局实例
ml_system = MLEnhancedPolyChat()