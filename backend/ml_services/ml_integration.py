cat > ml_integration.py << 'EOF'
"""
ML增强功能集成模块
"""
from typing import Dict, List, Optional
import asyncio
import time
from datetime import datetime

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
        
    async def process_enhanced_query(
        self, 
        prompt: str, 
        history: List = None,
        use_cache: bool = True,
        providers_to_use: Optional[List[str]] = None
    ) -> Dict:
        """增强版查询处理"""
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
                    'process_time': time.time() - start_time
                }
        
        # 2. 分类查询
        query_type = self.optimizer.classify_query(prompt)
        
        # 3. 选择API
        if not providers_to_use:
            selected_api = self.optimizer.select_api(query_type)
            providers_to_use = [selected_api]
        
        # 4. 调用providers
        responses = {}
        for provider_name in providers_to_use:
            if provider_name in self.providers:
                try:
                    provider = self.providers[provider_name]
                    response = await provider.generate_response(prompt, history or [])
                    responses[provider_name] = response
                except Exception as e:
                    print(f"Error with {provider_name}: {e}")
                    responses[provider_name] = f"Error: {str(e)}"
        
        # 5. 评分
        comparison_results = self.scorer.compare_responses(prompt, responses)
        
        # 6. 添加到缓存
        if use_cache:
            self.cache.add_to_cache(prompt, responses, comparison_results)
        
        # 7. 更新优化器
        for provider_name, result in comparison_results.items():
            success = result['scores']['overall'] > 0.6
            self.optimizer.update_performance(provider_name, success)
        
        return {
            'prompt': prompt,
            'query_type': query_type,
            'responses': comparison_results,
            'cache_hit': False,
            'process_time': time.time() - start_time
        }
    
    async def process_single_provider(
        self,
        provider: str,
        prompt: str,
        history: List = None
    ) -> Dict:
        """单provider处理（向后兼容）"""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        start_time = time.time()
        
        # 调用provider
        response = await self.providers[provider].generate_response(prompt, history or [])
        
        # 评分
        scores = self.scorer.score_response(prompt, response)
        
        return {
            'provider': provider,
            'response': response,
            'scores': scores,
            'response_time': time.time() - start_time
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'cache_stats': self.cache.get_cache_stats(),
            'api_stats': self.optimizer.get_api_stats()
        }

# 创建全局实例
ml_system = MLEnhancedPolyChat()
EOF