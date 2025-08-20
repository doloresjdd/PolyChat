"""简单缓存管理器（不需要faiss）"""
import pickle
import os
from typing import Dict, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SmartCacheManager:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print("Loading sentence encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []
        self.load_cache()
        
    def load_cache(self):
        cache_file = os.path.join(self.cache_dir, "cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.cache = pickle.load(f)
                
    def save_cache(self):
        cache_file = os.path.join(self.cache_dir, "cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        
    def add_to_cache(self, query: str, responses: Dict, scores: Dict = None):
        embedding = self.encoder.encode([query])[0]
        self.cache.append({
            'query': query,
            'embedding': embedding,
            'responses': responses,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        })
        self.save_cache()
        
    def search_similar(self, query: str, threshold=0.85) -> Optional[Dict]:
        if not self.cache:
            return None
            
        query_emb = self.encoder.encode([query])
        best_match = None
        best_similarity = 0
        
        for item in self.cache:
            similarity = cosine_similarity(
                query_emb.reshape(1, -1), 
                item['embedding'].reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = item
                
        if best_similarity > threshold:
            return {
                'responses': best_match['responses'],
                'scores': best_match['scores'],
                'similarity': float(best_similarity),
                'cached_query': best_match['query']
            }
        return None
        
    def get_cache_stats(self):
        return {
            'total_entries': len(self.cache),
            'cache_size': len(self.cache)
        }
