"""Caching implementation for Agent Skeptic Bench."""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import redis
from redis.asyncio import Redis as AsyncRedis

from .database.repositories import CacheRepository


logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching operations with multiple backends."""
    
    def __init__(self, 
                 redis_url: Optional[str] = None,
                 default_ttl: int = 3600,
                 use_database_fallback: bool = True):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            use_database_fallback: Whether to use database as fallback
        """
        self.default_ttl = default_ttl
        self.use_database_fallback = use_database_fallback
        
        # Redis setup
        self.redis_url = redis_url or self._get_redis_url()
        self.redis_client: Optional[AsyncRedis] = None
        self.redis_available = False
        
        # Memory cache as last resort
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Cache manager initialized with Redis URL: {self._mask_url(self.redis_url)}")
    
    def _get_redis_url(self) -> str:
        """Get Redis URL from environment variables."""
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            return redis_url
        
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        db = os.getenv("REDIS_DB", "0")
        password = os.getenv("REDIS_PASSWORD", "")
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in URL for logging."""
        if ":" in url and "@" in url:
            parts = url.split("@")
            if "//" in parts[0]:
                protocol_and_auth = parts[0].split("//")
                return f"{protocol_and_auth[0]}://***@{parts[1]}"
        return url
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = AsyncRedis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            self.redis_available = False
            self.redis_client = None
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self.redis_available = False
            logger.info("Redis cache connection closed")
    
    def _generate_key(self, prefix: str, *args: Any) -> str:
        """Generate cache key from prefix and arguments."""
        key_parts = [prefix] + [str(arg) for arg in args]
        key = ":".join(key_parts)
        
        # Hash if key is too long
        if len(key) > 250:
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try Redis first
        if self.redis_available and self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed for key {key}: {e}")
        
        # Try memory cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if "expires_at" not in entry or datetime.utcnow() < entry["expires_at"]:
                return entry["value"]
            else:
                del self._memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        success = False
        
        # Try Redis first
        if self.redis_available and self.redis_client:
            try:
                serialized_value = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized_value)
                success = True
            except Exception as e:
                logger.warning(f"Redis set failed for key {key}: {e}")
        
        # Always store in memory cache as backup
        self._memory_cache[key] = {
            "value": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
        }
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        success = False
        
        # Try Redis first
        if self.redis_available and self.redis_client:
            try:
                result = await self.redis_client.delete(key)
                success = result > 0
            except Exception as e:
                logger.warning(f"Redis delete failed for key {key}: {e}")
        
        # Remove from memory cache
        if key in self._memory_cache:
            del self._memory_cache[key]
            success = True
        
        return success
    
    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix."""
        deleted_count = 0
        
        # Clear from Redis
        if self.redis_available and self.redis_client:
            try:
                keys = await self.redis_client.keys(f"{prefix}*")
                if keys:
                    deleted_count = await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear prefix failed for {prefix}: {e}")
        
        # Clear from memory cache
        memory_keys = [k for k in self._memory_cache.keys() if k.startswith(prefix)]
        for key in memory_keys:
            del self._memory_cache[key]
        
        deleted_count += len(memory_keys)
        return deleted_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health."""
        status = {
            "redis_available": False,
            "memory_cache_size": len(self._memory_cache),
            "total_backends": 1  # memory cache always available
        }
        
        if self.redis_client:
            try:
                await self.redis_client.ping()
                status["redis_available"] = True
                status["total_backends"] += 1
                
                # Get Redis info
                info = await self.redis_client.info()
                status["redis_memory_used"] = info.get("used_memory_human", "unknown")
                status["redis_connected_clients"] = info.get("connected_clients", 0)
                
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
        
        return status
    
    def _cleanup_memory_cache(self) -> None:
        """Clean up expired entries from memory cache."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if "expires_at" in entry and now > entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")


class ScenarioCache:
    """Specialized cache for scenarios."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "scenario"
    
    async def get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get cached scenario."""
        key = self.cache_manager._generate_key(self.prefix, scenario_id)
        return await self.cache_manager.get(key)
    
    async def set_scenario(self, scenario_id: str, scenario_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache scenario data."""
        key = self.cache_manager._generate_key(self.prefix, scenario_id)
        return await self.cache_manager.set(key, scenario_data, ttl)
    
    async def get_category_scenarios(self, category: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached scenarios for a category."""
        key = self.cache_manager._generate_key(self.prefix, "category", category)
        return await self.cache_manager.get(key)
    
    async def set_category_scenarios(self, category: str, scenarios: List[Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """Cache scenarios for a category."""
        key = self.cache_manager._generate_key(self.prefix, "category", category)
        return await self.cache_manager.set(key, scenarios, ttl)
    
    async def invalidate_scenario(self, scenario_id: str) -> bool:
        """Invalidate scenario cache."""
        key = self.cache_manager._generate_key(self.prefix, scenario_id)
        return await self.cache_manager.delete(key)
    
    async def invalidate_category(self, category: str) -> bool:
        """Invalidate category cache."""
        key = self.cache_manager._generate_key(self.prefix, "category", category)
        return await self.cache_manager.delete(key)


class EvaluationCache:
    """Specialized cache for evaluation results."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "evaluation"
    
    async def get_result(self, agent_id: str, scenario_id: str, config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached evaluation result."""
        key = self.cache_manager._generate_key(self.prefix, agent_id, scenario_id, config_hash)
        return await self.cache_manager.get(key)
    
    async def set_result(self, agent_id: str, scenario_id: str, config_hash: str, 
                        result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache evaluation result."""
        key = self.cache_manager._generate_key(self.prefix, agent_id, scenario_id, config_hash)
        return await self.cache_manager.set(key, result, ttl)
    
    def generate_config_hash(self, agent_config: Dict[str, Any]) -> str:
        """Generate hash for agent configuration."""
        # Remove sensitive fields
        config_copy = {k: v for k, v in agent_config.items() if k != "api_key"}
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
    
    return _cache_manager


async def initialize_cache() -> None:
    """Initialize global cache manager."""
    cache_manager = get_cache_manager()
    await cache_manager.initialize()


async def close_cache() -> None:
    """Close global cache manager."""
    global _cache_manager
    
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None