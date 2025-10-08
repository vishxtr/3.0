# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Redis caching and session management
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import hashlib

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis
import pickle

from config import get_config, RedisConfig

logger = logging.getLogger(__name__)

class RedisManager:
    """Redis manager for caching and session management"""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or get_config("redis")
        self.redis_client = None
        self.connection_pool = None
        self.is_initialized = False
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options
            )
            
            # Create Redis client
            self.redis_client = Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            # Configure Redis settings
            await self._configure_redis()
            
            self.is_initialized = True
            logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
            raise
    
    async def _configure_redis(self):
        """Configure Redis settings"""
        try:
            # Set memory policy
            await self.redis_client.config_set("maxmemory", self.config.max_memory)
            await self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            
            # Enable keyspace notifications for cache invalidation
            await self.redis_client.config_set("notify-keyspace-events", "Ex")
            
            logger.info("Redis configuration applied successfully")
            
        except Exception as e:
            logger.warning(f"Failed to configure Redis settings: {e}")
    
    async def close(self):
        """Close Redis connection"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            self.is_initialized = False
            logger.info("Redis manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing Redis manager: {e}")
    
    def is_healthy(self) -> bool:
        """Check if Redis is healthy"""
        return self.is_initialized and self.redis_client is not None
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Generate cache key for content"""
        # Create hash of content for consistent key generation
        content_hash = hashlib.md5(f"{content_type}:{content}".encode()).hexdigest()
        return f"cache:{content_type}:{content_hash}"
    
    def _generate_session_key(self, session_id: str) -> str:
        """Generate session key"""
        return f"{self.config.session_prefix}{session_id}"
    
    def _generate_queue_key(self, queue_name: str) -> str:
        """Generate queue key"""
        return f"{self.config.queue_prefix}{queue_name}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            # Try JSON first for simple data types
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                return json.dumps(data).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(data)
        except Exception as e:
            logger.warning(f"Serialization failed, using pickle: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(data)
            except Exception as e:
                logger.error(f"Deserialization failed: {e}")
                return None
    
    # Cache operations
    async def get_cached_result(self, content: str, content_type: str) -> Optional[Dict[str, Any]]:
        """Get cached inference result"""
        try:
            if not self.is_healthy():
                return None
            
            cache_key = self._generate_cache_key(content, content_type)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.cache_stats["hits"] += 1
                result = self._deserialize_data(cached_data)
                
                # Check if result is still valid
                if result and self._is_result_valid(result):
                    logger.debug(f"Cache hit for {content_type}: {content[:50]}...")
                    return result
                else:
                    # Remove expired result
                    await self.redis_client.delete(cache_key)
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    async def cache_result(
        self, 
        content: str, 
        content_type: str, 
        result: Dict[str, Any], 
        ttl: Optional[int] = None
    ):
        """Cache inference result"""
        try:
            if not self.is_healthy():
                return False
            
            cache_key = self._generate_cache_key(content, content_type)
            ttl = ttl or self.config.default_ttl
            
            # Add metadata to result
            result["cached_at"] = datetime.utcnow().isoformat()
            result["cache_ttl"] = ttl
            
            # Serialize and store
            serialized_data = self._serialize_data(result)
            await self.redis_client.setex(cache_key, ttl, serialized_data)
            
            self.cache_stats["sets"] += 1
            logger.debug(f"Cached result for {content_type}: {content[:50]}...")
            return True
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Failed to cache result: {e}")
            return False
    
    async def invalidate_cache(self, content: str, content_type: str):
        """Invalidate cached result"""
        try:
            if not self.is_healthy():
                return False
            
            cache_key = self._generate_cache_key(content, content_type)
            await self.redis_client.delete(cache_key)
            
            self.cache_stats["deletes"] += 1
            logger.debug(f"Invalidated cache for {content_type}: {content[:50]}...")
            return True
            
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    async def clear_cache(self, pattern: str = "cache:*"):
        """Clear all cache entries"""
        try:
            if not self.is_healthy():
                return False
            
            # Get all cache keys
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            
            logger.info(f"Cleared {len(keys)} cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def _is_result_valid(self, result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid"""
        try:
            cached_at = result.get("cached_at")
            cache_ttl = result.get("cache_ttl", self.config.default_ttl)
            
            if not cached_at:
                return False
            
            # Parse cached timestamp
            cached_time = datetime.fromisoformat(cached_at.replace('Z', '+00:00'))
            expiry_time = cached_time + timedelta(seconds=cache_ttl)
            
            return datetime.utcnow() < expiry_time
            
        except Exception as e:
            logger.warning(f"Failed to validate cached result: {e}")
            return False
    
    # Session management
    async def create_session(self, session_data: Dict[str, Any], ttl: Optional[int] = None) -> str:
        """Create a new session"""
        try:
            if not self.is_healthy():
                raise Exception("Redis not available")
            
            session_id = f"session_{int(time.time())}_{hash(str(session_data))}"
            session_key = self._generate_session_key(session_id)
            ttl = ttl or self.config.session_ttl
            
            # Add session metadata
            session_data["created_at"] = datetime.utcnow().isoformat()
            session_data["last_accessed"] = datetime.utcnow().isoformat()
            
            # Store session
            serialized_data = self._serialize_data(session_data)
            await self.redis_client.setex(session_key, ttl, serialized_data)
            
            logger.debug(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            if not self.is_healthy():
                return None
            
            session_key = self._generate_session_key(session_id)
            session_data = await self.redis_client.get(session_key)
            
            if session_data:
                result = self._deserialize_data(session_data)
                
                # Update last accessed time
                if result:
                    result["last_accessed"] = datetime.utcnow().isoformat()
                    await self.update_session(session_id, result)
                
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, session_data: Dict[str, Any], ttl: Optional[int] = None):
        """Update session data"""
        try:
            if not self.is_healthy():
                return False
            
            session_key = self._generate_session_key(session_id)
            ttl = ttl or self.config.session_ttl
            
            # Update metadata
            session_data["last_accessed"] = datetime.utcnow().isoformat()
            
            # Store updated session
            serialized_data = self._serialize_data(session_data)
            await self.redis_client.setex(session_key, ttl, serialized_data)
            
            logger.debug(f"Updated session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str):
        """Delete session"""
        try:
            if not self.is_healthy():
                return False
            
            session_key = self._generate_session_key(session_id)
            await self.redis_client.delete(session_key)
            
            logger.debug(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def cleanup_expired_sessions(self):
        """Cleanup expired sessions"""
        try:
            if not self.is_healthy():
                return 0
            
            # Redis TTL handles expiration automatically
            # This method can be used for additional cleanup if needed
            pattern = f"{self.config.session_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_count += 1
            
            logger.info(f"Found {expired_count} expired sessions")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    # Queue operations
    async def enqueue_request(self, queue_name: str, request_data: Dict[str, Any], priority: int = 1):
        """Enqueue a request for processing"""
        try:
            if not self.is_healthy():
                return False
            
            queue_key = self._generate_queue_key(queue_name)
            
            # Add request metadata
            request_data["enqueued_at"] = datetime.utcnow().isoformat()
            request_data["priority"] = priority
            
            # Serialize and enqueue
            serialized_data = self._serialize_data(request_data)
            
            # Use priority-based queuing
            if priority > 3:  # High priority
                await self.redis_client.lpush(f"{queue_key}:high", serialized_data)
            elif priority > 1:  # Medium priority
                await self.redis_client.lpush(f"{queue_key}:medium", serialized_data)
            else:  # Low priority
                await self.redis_client.lpush(f"{queue_key}:low", serialized_data)
            
            logger.debug(f"Enqueued request in {queue_name} with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue request: {e}")
            return False
    
    async def dequeue_request(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Dequeue a request for processing"""
        try:
            if not self.is_healthy():
                return None
            
            queue_key = self._generate_queue_key(queue_name)
            
            # Try high priority first, then medium, then low
            for priority in ["high", "medium", "low"]:
                priority_queue = f"{queue_key}:{priority}"
                request_data = await self.redis_client.rpop(priority_queue)
                
                if request_data:
                    result = self._deserialize_data(request_data)
                    logger.debug(f"Dequeued {priority} priority request from {queue_name}")
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue request: {e}")
            return None
    
    async def get_queue_length(self, queue_name: str) -> int:
        """Get queue length"""
        try:
            if not self.is_healthy():
                return 0
            
            queue_key = self._generate_queue_key(queue_name)
            total_length = 0
            
            # Sum all priority queues
            for priority in ["high", "medium", "low"]:
                priority_queue = f"{queue_key}:{priority}"
                length = await self.redis_client.llen(priority_queue)
                total_length += length
            
            return total_length
            
        except Exception as e:
            logger.error(f"Failed to get queue length: {e}")
            return 0
    
    # Request status tracking
    async def set_request_status(self, request_id: str, status_data: Dict[str, Any], ttl: Optional[int] = None):
        """Set request status"""
        try:
            if not self.is_healthy():
                return False
            
            status_key = f"status:{request_id}"
            ttl = ttl or self.config.result_ttl
            
            # Add timestamp
            status_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Store status
            serialized_data = self._serialize_data(status_data)
            await self.redis_client.setex(status_key, ttl, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set request status: {e}")
            return False
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request status"""
        try:
            if not self.is_healthy():
                return None
            
            status_key = f"status:{request_id}"
            status_data = await self.redis_client.get(status_key)
            
            if status_data:
                return self._deserialize_data(status_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get request status: {e}")
            return None
    
    # Statistics and monitoring
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.is_healthy():
                return {}
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Calculate cache hit rate
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "cache_hits": self.cache_stats["hits"],
                "cache_misses": self.cache_stats["misses"],
                "cache_sets": self.cache_stats["sets"],
                "cache_deletes": self.cache_stats["deletes"],
                "cache_errors": self.cache_stats["errors"],
                "cache_hit_rate": hit_rate,
                "redis_memory_used": info.get("used_memory_human", "0B"),
                "redis_connected_clients": info.get("connected_clients", 0),
                "redis_ops_per_sec": info.get("instantaneous_ops_per_sec", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            if not self.is_healthy():
                return {}
            
            info = await self.redis_client.info()
            return {
                "version": info.get("redis_version", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0),
                "memory_used": info.get("used_memory_human", "0B"),
                "memory_peak": info.get("used_memory_peak_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    # Pub/Sub operations
    async def publish_message(self, channel: str, message: Dict[str, Any]):
        """Publish message to channel"""
        try:
            if not self.is_healthy():
                return False
            
            channel_key = f"{self.config.pubsub_prefix}{channel}"
            serialized_message = self._serialize_data(message)
            
            await self.redis_client.publish(channel_key, serialized_message)
            logger.debug(f"Published message to channel: {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe_to_channel(self, channel: str, callback):
        """Subscribe to channel and process messages"""
        try:
            if not self.is_healthy():
                return False
            
            channel_key = f"{self.config.pubsub_prefix}{channel}"
            pubsub = self.redis_client.pubsub()
            
            await pubsub.subscribe(channel_key)
            logger.info(f"Subscribed to channel: {channel}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = self._deserialize_data(message["data"])
                        await callback(channel, data)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")
            return False

def create_redis_manager(config: Optional[RedisConfig] = None) -> RedisManager:
    """Factory function to create Redis manager instance"""
    return RedisManager(config)