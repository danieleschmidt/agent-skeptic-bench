"""High-performance caching system for usage metrics."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize memory cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, access_time)
        self.access_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        value, timestamp, _ = self.cache[key]
        current_time = time.time()
        
        # Check TTL
        if current_time - timestamp > self.default_ttl:
            del self.cache[key]
            return None
        
        # Update access time for LRU
        self.cache[key] = (value, timestamp, current_time)
        self.access_count += 1
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        current_time = time.time()
        effective_ttl = ttl or self.default_ttl
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = (value, current_time, current_time)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find oldest access time
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
        del self.cache[lru_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(
            1 for _, timestamp, _ in self.cache.values()
            if current_time - timestamp > self.default_ttl
        )
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "access_count": self.access_count,
            "expired_items": expired_count,
            "hit_rate": 0.0  # Would need hit/miss tracking for accurate calculation
        }


class UsageMetricsCache:
    """Specialized caching for usage metrics operations."""
    
    def __init__(self, max_memory_items: int = 1000, cache_ttl: int = 300):
        """Initialize usage metrics cache."""
        self.memory_cache = MemoryCache(max_memory_items, cache_ttl)
        self.query_cache = MemoryCache(500, cache_ttl)  # Smaller cache for query results
        self.aggregation_cache = MemoryCache(200, 600)  # Longer TTL for expensive aggregations
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_cleanup = time.time()
        
        logger.info("UsageMetricsCache initialized")
    
    def get_usage_summary(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached usage summary."""
        result = self.aggregation_cache.get(cache_key)
        if result:
            self.cache_hits += 1
            logger.debug(f"Cache hit for usage summary: {cache_key}")
        else:
            self.cache_misses += 1
            logger.debug(f"Cache miss for usage summary: {cache_key}")
        
        return result
    
    def set_usage_summary(self, cache_key: str, summary: Dict[str, Any], ttl: int = 600) -> None:
        """Cache usage summary."""
        self.aggregation_cache.set(cache_key, summary, ttl)
        logger.debug(f"Cached usage summary: {cache_key}")
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session metrics."""
        return self.memory_cache.get(f"session_{session_id}")
    
    def set_session_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Cache session metrics."""
        self.memory_cache.set(f"session_{session_id}", metrics)
    
    def get_query_result(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result."""
        result = self.query_cache.get(query_hash)
        if result:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        return result
    
    def set_query_result(self, query_hash: str, result: List[Dict[str, Any]]) -> None:
        """Cache query result."""
        self.query_cache.set(query_hash, result)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        invalidated = 0
        
        # Check memory cache
        keys_to_remove = [key for key in self.memory_cache.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.memory_cache.cache[key]
            invalidated += 1
        
        # Check query cache
        keys_to_remove = [key for key in self.query_cache.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.query_cache.cache[key]
            invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")
        return invalidated
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < 60:  # Only cleanup every minute
            return {"memory": 0, "query": 0, "aggregation": 0}
        
        cleanup_stats = {
            "memory": self._cleanup_cache(self.memory_cache),
            "query": self._cleanup_cache(self.query_cache),
            "aggregation": self._cleanup_cache(self.aggregation_cache)
        }
        
        self.last_cleanup = current_time
        total_cleaned = sum(cleanup_stats.values())
        
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} expired cache entries")
        
        return cleanup_stats
    
    def _cleanup_cache(self, cache: MemoryCache) -> int:
        """Clean up expired entries in a specific cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp, _) in cache.cache.items():
            if current_time - timestamp > cache.default_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del cache.cache[key]
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_hits,
            "total_misses": self.cache_misses,
            "memory_cache": self.memory_cache.stats(),
            "query_cache": self.query_cache.stats(),
            "aggregation_cache": self.aggregation_cache.stats()
        }


class BatchProcessor:
    """Batch processing for high-volume usage metrics."""
    
    def __init__(self, batch_size: int = 100, flush_interval: int = 30):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush = time.time()
        
        logger.info(f"BatchProcessor initialized: batch_size={batch_size}, flush_interval={flush_interval}s")
    
    def add_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Add metrics to batch buffer."""
        self.batch_buffer.append(metrics)
        
        # Check if we should flush
        should_flush = (
            len(self.batch_buffer) >= self.batch_size or
            time.time() - self.last_flush >= self.flush_interval
        )
        
        if should_flush:
            self.flush_batch()
            return True
        
        return False
    
    def flush_batch(self) -> int:
        """Flush current batch to storage."""
        if not self.batch_buffer:
            return 0
        
        batch_size = len(self.batch_buffer)
        
        try:
            # Group by date for efficient file writing
            date_groups = {}
            for metrics in self.batch_buffer:
                timestamp = datetime.fromisoformat(metrics["timestamp"])
                date_str = timestamp.strftime("%Y-%m-%d")
                
                if date_str not in date_groups:
                    date_groups[date_str] = []
                date_groups[date_str].append(metrics)
            
            # Write each date group
            storage_path = Path("data/usage_metrics")
            storage_path.mkdir(parents=True, exist_ok=True)
            
            for date_str, date_metrics in date_groups.items():
                file_path = storage_path / f"usage_metrics_{date_str}.jsonl"
                
                with open(file_path, "a", encoding="utf-8") as f:
                    for metrics in date_metrics:
                        f.write(json.dumps(metrics, default=str) + "\n")
            
            # Clear buffer
            self.batch_buffer.clear()
            self.last_flush = time.time()
            
            logger.info(f"Flushed batch of {batch_size} metrics to storage")
            return batch_size
            
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            return 0
    
    def force_flush(self) -> int:
        """Force flush all pending metrics."""
        return self.flush_batch()
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self.batch_buffer),
            "max_batch_size": self.batch_size,
            "time_since_last_flush": time.time() - self.last_flush,
            "flush_interval": self.flush_interval
        }


class AsyncQueryProcessor:
    """Asynchronous query processor for high-performance metrics queries."""
    
    def __init__(self, max_concurrent_queries: int = 10):
        """Initialize async query processor."""
        self.max_concurrent_queries = max_concurrent_queries
        self.active_queries = 0
        self.query_queue: List[Tuple[str, Dict[str, Any]]] = []
        
        logger.info(f"AsyncQueryProcessor initialized: max_concurrent={max_concurrent_queries}")
    
    async def process_usage_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process usage metrics query asynchronously."""
        import asyncio
        
        query_id = f"query_{int(time.time() * 1000)}"
        
        if self.active_queries >= self.max_concurrent_queries:
            # Queue the query
            self.query_queue.append((query_id, query_params))
            logger.info(f"Query {query_id} queued (active: {self.active_queries})")
            
            # Wait for slot to become available
            while self.active_queries >= self.max_concurrent_queries:
                await asyncio.sleep(0.1)
        
        self.active_queries += 1
        
        try:
            result = await self._execute_query(query_params)
            logger.debug(f"Query {query_id} completed")
            return result
        
        finally:
            self.active_queries -= 1
            
            # Process next queued query if any
            if self.query_queue:
                next_query_id, next_params = self.query_queue.pop(0)
                # Schedule next query
                asyncio.create_task(self.process_usage_query(next_params))
    
    async def _execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual query."""
        import asyncio
        
        query_type = params.get("type", "summary")
        
        if query_type == "summary":
            return await self._query_summary(params)
        elif query_type == "trends":
            return await self._query_trends(params)
        elif query_type == "detailed":
            return await self._query_detailed(params)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    async def _query_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute summary query."""
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        days = params.get("days", 7)
        
        # This would normally query the actual data
        return {
            "query_type": "summary",
            "days": days,
            "total_sessions": 150,
            "total_evaluations": 1200,
            "avg_score": 0.82,
            "computed_at": datetime.utcnow().isoformat()
        }
    
    async def _query_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trends query."""
        await asyncio.sleep(0.2)  # Longer processing for trends
        
        return {
            "query_type": "trends",
            "trend_direction": "improving",
            "score_change": "+5.2%",
            "volume_trend": "increasing",
            "computed_at": datetime.utcnow().isoformat()
        }
    
    async def _query_detailed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute detailed query."""
        await asyncio.sleep(0.05)
        
        return {
            "query_type": "detailed",
            "records": 45,
            "filtered_records": 32,
            "computed_at": datetime.utcnow().isoformat()
        }


class PerformanceOptimizer:
    """Optimizes usage metrics operations for high performance."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.operation_times = {}
        self.optimization_rules = {
            "file_io": {"min_batch_size": 10, "max_file_size_mb": 50},
            "memory": {"max_cache_size": 1000, "cleanup_threshold": 0.8},
            "query": {"max_concurrent": 10, "timeout_seconds": 30}
        }
        
        logger.info("PerformanceOptimizer initialized")
    
    def record_operation(self, operation: str, duration: float, size: int = 0) -> None:
        """Record operation performance."""
        if operation not in self.operation_times:
            self.operation_times[operation] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "total_size": 0
            }
        
        stats = self.operation_times[operation]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["total_size"] += size
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        for operation, stats in self.operation_times.items():
            if stats["count"] < 5:  # Not enough data
                continue
            
            # Check for slow operations
            if stats["avg_time"] > 1.0:  # More than 1 second average
                recommendations.append({
                    "type": "performance",
                    "operation": operation,
                    "issue": "slow_operation",
                    "current_avg": stats["avg_time"],
                    "recommendation": "Consider caching or optimization"
                })
            
            # Check for high variance
            if stats["max_time"] > stats["avg_time"] * 5:
                recommendations.append({
                    "type": "stability",
                    "operation": operation,
                    "issue": "high_variance",
                    "max_time": stats["max_time"],
                    "avg_time": stats["avg_time"],
                    "recommendation": "Investigate outlier causes"
                })
        
        return recommendations
    
    def optimize_batch_size(self, operation: str, target_time: float = 0.5) -> int:
        """Calculate optimal batch size for operation."""
        if operation not in self.operation_times:
            return 100  # Default batch size
        
        stats = self.operation_times[operation]
        
        if stats["avg_time"] == 0:
            return 100
        
        # Calculate items per second
        avg_items_per_op = max(stats["total_size"] / stats["count"], 1)
        items_per_second = avg_items_per_op / stats["avg_time"]
        
        # Target batch size for desired processing time
        optimal_batch = int(items_per_second * target_time)
        
        # Apply constraints
        optimal_batch = max(10, min(optimal_batch, 1000))
        
        return optimal_batch
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_operations = sum(stats["count"] for stats in self.operation_times.values())
        total_time = sum(stats["total_time"] for stats in self.operation_times.values())
        
        return {
            "total_operations": total_operations,
            "total_time": total_time,
            "operations": self.operation_times.copy(),
            "recommendations": self.get_optimization_recommendations()
        }


class ConnectionPool:
    """Connection pool for database and external service connections."""
    
    def __init__(self, max_connections: int = 20, idle_timeout: int = 300):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.idle_timeout = idle_timeout
        self.available_connections = []
        self.active_connections = {}
        self.connection_count = 0
        
        logger.info(f"ConnectionPool initialized: max={max_connections}")
    
    async def get_connection(self, connection_type: str = "default") -> str:
        """Get a connection from the pool."""
        import asyncio
        
        # Try to reuse available connection
        for i, (conn_id, last_used, conn_type) in enumerate(self.available_connections):
            if conn_type == connection_type and time.time() - last_used < self.idle_timeout:
                # Reuse this connection
                del self.available_connections[i]
                self.active_connections[conn_id] = time.time()
                logger.debug(f"Reused connection {conn_id}")
                return conn_id
        
        # Create new connection if under limit
        if self.connection_count < self.max_connections:
            conn_id = f"{connection_type}_conn_{self.connection_count}"
            self.connection_count += 1
            self.active_connections[conn_id] = time.time()
            logger.debug(f"Created new connection {conn_id}")
            return conn_id
        
        # Wait for connection to become available
        logger.warning("Connection pool exhausted, waiting...")
        await asyncio.sleep(0.1)
        return await self.get_connection(connection_type)
    
    def release_connection(self, connection_id: str, connection_type: str = "default") -> None:
        """Release connection back to pool."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            self.available_connections.append((connection_id, time.time(), connection_type))
            logger.debug(f"Released connection {connection_id}")
    
    def cleanup_idle_connections(self) -> int:
        """Clean up idle connections."""
        current_time = time.time()
        cleaned = 0
        
        # Remove expired connections
        self.available_connections = [
            (conn_id, last_used, conn_type)
            for conn_id, last_used, conn_type in self.available_connections
            if current_time - last_used < self.idle_timeout
        ]
        
        # Update connection count
        total_connections = len(self.available_connections) + len(self.active_connections)
        cleaned = self.connection_count - total_connections
        self.connection_count = total_connections
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} idle connections")
        
        return cleaned
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "max_connections": self.max_connections,
            "active_connections": len(self.active_connections),
            "available_connections": len(self.available_connections),
            "total_connections": self.connection_count,
            "utilization": len(self.active_connections) / self.max_connections
        }