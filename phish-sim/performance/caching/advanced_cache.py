# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Advanced Caching System for Performance Optimization
Implements multi-level caching with intelligent invalidation and prefetching
"""

import json
import hashlib
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
except ImportError:
    redis = None
    Redis = None

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels for different types of data"""
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_REDIS = "l2_redis"        # Redis cache (fast)
    L3_PERSISTENT = "l3_persistent"  # Persistent storage (slower)

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    TTL = "ttl"                  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int] = None
    level: CacheLevel = CacheLevel.L2_REDIS
    strategy: CacheStrategy = CacheStrategy.TTL
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > (self.created_at + timedelta(seconds=self.ttl))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl': self.ttl,
            'level': self.level.value,
            'strategy': self.strategy.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(
            key=data['key'],
            value=data['value'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            ttl=data.get('ttl'),
            level=CacheLevel(data['level']),
            strategy=CacheStrategy(data['strategy'])
        )

class AdvancedCache:
    """Advanced multi-level caching system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 max_memory_entries: int = 1000,
                 default_ttl: int = 300):
        self.redis_url = redis_url
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        
        # L1 Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_access_order: List[str] = []
        
        # Redis connection
        self.redis_client: Optional[Redis] = None
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'writes': 0,
            'deletes': 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            'avg_get_time': 0.0,
            'avg_set_time': 0.0,
            'cache_hit_ratio': 0.0
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        if redis is None:
            logger.warning("Redis not available, using memory-only cache")
            return
        
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"{prefix}:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level fallback"""
        start_time = time.time()
        
        try:
            # L1 Memory cache check
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.last_accessed = datetime.utcnow()
                    entry.access_count += 1
                    self._update_memory_access_order(key)
                    self.stats['hits'] += 1
                    self._update_performance_metrics('get', time.time() - start_time)
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
                    self.memory_access_order.remove(key)
            
            # L2 Redis cache check
            if self.redis_client:
                try:
                    cached_data = await self.redis_client.get(key)
                    if cached_data:
                        entry_data = json.loads(cached_data)
                        entry = CacheEntry.from_dict(entry_data)
                        if not entry.is_expired():
                            # Promote to L1 cache
                            await self._promote_to_memory(entry)
                            self.stats['hits'] += 1
                            self._update_performance_metrics('get', time.time() - start_time)
                            return entry.value
                        else:
                            # Remove expired entry
                            await self.redis_client.delete(key)
                except Exception as e:
                    logger.error(f"Redis get error: {e}")
            
            # Cache miss
            self.stats['misses'] += 1
            self._update_performance_metrics('get', time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats['misses'] += 1
            return default
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  level: CacheLevel = CacheLevel.L2_REDIS,
                  strategy: CacheStrategy = CacheStrategy.TTL) -> bool:
        """Set value in cache with specified level and strategy"""
        start_time = time.time()
        
        try:
            ttl = ttl or self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                ttl=ttl,
                level=level,
                strategy=strategy
            )
            
            # Set in appropriate cache level
            if level == CacheLevel.L1_MEMORY:
                await self._set_memory(entry)
            elif level == CacheLevel.L2_REDIS and self.redis_client:
                await self._set_redis(entry)
            elif level == CacheLevel.L3_PERSISTENT:
                await self._set_persistent(entry)
            
            self.stats['writes'] += 1
            self._update_performance_metrics('set', time.time() - start_time)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _set_memory(self, entry: CacheEntry):
        """Set entry in memory cache"""
        # Check if we need to evict
        if len(self.memory_cache) >= self.max_memory_entries:
            await self._evict_memory_entry()
        
        self.memory_cache[entry.key] = entry
        self._update_memory_access_order(entry.key)
    
    async def _set_redis(self, entry: CacheEntry):
        """Set entry in Redis cache"""
        if self.redis_client:
            entry_data = json.dumps(entry.to_dict())
            if entry.ttl:
                await self.redis_client.setex(entry.key, entry.ttl, entry_data)
            else:
                await self.redis_client.set(entry.key, entry_data)
    
    async def _set_persistent(self, entry: CacheEntry):
        """Set entry in persistent storage (placeholder)"""
        # This would implement persistent storage like file system or database
        logger.info(f"Persistent storage not implemented for key: {entry.key}")
    
    async def _promote_to_memory(self, entry: CacheEntry):
        """Promote Redis entry to memory cache"""
        if len(self.memory_cache) >= self.max_memory_entries:
            await self._evict_memory_entry()
        
        self.memory_cache[entry.key] = entry
        self._update_memory_access_order(entry.key)
    
    async def _evict_memory_entry(self):
        """Evict entry from memory cache using LRU strategy"""
        if not self.memory_access_order:
            return
        
        # Remove least recently used entry
        lru_key = self.memory_access_order[0]
        if lru_key in self.memory_cache:
            del self.memory_cache[lru_key]
            self.memory_access_order.remove(lru_key)
            self.stats['evictions'] += 1
    
    def _update_memory_access_order(self, key: str):
        """Update memory access order for LRU"""
        if key in self.memory_access_order:
            self.memory_access_order.remove(key)
        self.memory_access_order.append(key)
    
    async def delete(self, key: str) -> bool:
        """Delete entry from all cache levels"""
        try:
            # Remove from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
                if key in self.memory_access_order:
                    self.memory_access_order.remove(key)
            
            # Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
            
            self.stats['deletes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self, level: Optional[CacheLevel] = None):
        """Clear cache at specified level or all levels"""
        try:
            if level is None or level == CacheLevel.L1_MEMORY:
                self.memory_cache.clear()
                self.memory_access_order.clear()
            
            if (level is None or level == CacheLevel.L2_REDIS) and self.redis_client:
                await self.redis_client.flushdb()
            
            logger.info(f"Cache cleared at level: {level or 'all'}")
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def _update_performance_metrics(self, operation: str, duration: float):
        """Update performance metrics"""
        if operation == 'get':
            # Update average get time
            total_ops = self.stats['hits'] + self.stats['misses']
            if total_ops > 0:
                self.performance_metrics['avg_get_time'] = (
                    (self.performance_metrics['avg_get_time'] * (total_ops - 1) + duration) / total_ops
                )
        elif operation == 'set':
            # Update average set time
            if self.stats['writes'] > 0:
                self.performance_metrics['avg_set_time'] = (
                    (self.performance_metrics['avg_set_time'] * (self.stats['writes'] - 1) + duration) / self.stats['writes']
                )
        
        # Update cache hit ratio
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            self.performance_metrics['cache_hit_ratio'] = self.stats['hits'] / total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics and performance metrics"""
        return {
            'stats': self.stats.copy(),
            'performance': self.performance_metrics.copy(),
            'memory_usage': {
                'entries': len(self.memory_cache),
                'max_entries': self.max_memory_entries,
                'usage_ratio': len(self.memory_cache) / self.max_memory_entries
            },
            'redis_connected': self.redis_client is not None
        }
    
    async def prefetch(self, keys: List[str], fetch_func, *args, **kwargs):
        """Prefetch multiple keys using provided fetch function"""
        tasks = []
        for key in keys:
            if await self.get(key) is None:
                tasks.append(self._prefetch_single(key, fetch_func, *args, **kwargs))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _prefetch_single(self, key: str, fetch_func, *args, **kwargs):
        """Prefetch single key"""
        try:
            value = await fetch_func(*args, **kwargs)
            await self.set(key, value)
        except Exception as e:
            logger.error(f"Prefetch error for key {key}: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        try:
            # Memory cache
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_access_order:
                    self.memory_access_order.remove(key)
            
            # Redis cache
            if self.redis_client:
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            
            logger.info(f"Invalidated {len(keys_to_remove)} keys matching pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Pattern invalidation error: {e}")

# Global cache instance
cache_instance: Optional[AdvancedCache] = None

async def get_cache() -> AdvancedCache:
    """Get global cache instance"""
    global cache_instance
    if cache_instance is None:
        cache_instance = AdvancedCache()
        await cache_instance.initialize()
    return cache_instance

# Cache decorators
def cached(ttl: int = 300, level: CacheLevel = CacheLevel.L2_REDIS, 
           key_prefix: str = "func"):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = await get_cache()
            key = cache._generate_key(key_prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache.set(key, result, ttl=ttl, level=level)
            return result
        
        return wrapper
    return decorator

def cache_invalidate(pattern: str = None):
    """Decorator for invalidating cache after function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            cache = await get_cache()
            if pattern:
                await cache.invalidate_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator