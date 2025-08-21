"""Performance Optimization Framework for Agent Skeptic Bench.

Advanced performance optimization including:
- Multi-level caching system
- Async processing and concurrency
- Resource pooling and optimization
- Auto-scaling triggers
- Performance monitoring and profiling
"""

import asyncio
import functools
import hashlib
import logging
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    cache_hit_rate: float
    throughput: float
    latency_p95: float
    timestamp: float = field(default_factory=time.time)


class LRUCache:
    """High-performance LRU cache with statistics."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Any] = {}
        self.access_order: deque = deque()
        self.timestamps: Dict[str, float] = {}
        self.stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamps.get(key, 0) > self.ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                self.stats.hits += 1
                self.stats.update_hit_rate()
                return self.cache[key]
            
            # Cache miss or expired
            if key in self.cache:
                self._remove(key)
            
            self.stats.misses += 1
            self.stats.update_hit_rate()
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                self._remove(key)
            
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Add new entry
            self.cache[key] = value
            self.access_order.append(key)
            self.timestamps[key] = time.time()
            
            self.stats.size = len(self.cache)
    
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
            del self.timestamps[key]
            self.stats.size = len(self.cache)
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if self.access_order:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            self.stats.evictions += 1
            self.stats.size = len(self.cache)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.timestamps.clear()
            self.stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class MultiLevelCache:
    """Multi-level cache system (L1: Memory, L2: Persistent)."""
    
    def __init__(self, l1_size: int = 1000, l2_size: int = 10000):
        """Initialize multi-level cache."""
        self.l1_cache = LRUCache(max_size=l1_size, ttl=300)  # 5 min TTL
        self.l2_cache = LRUCache(max_size=l2_size, ttl=3600)  # 1 hour TTL
        self.promotion_threshold = 2  # Promote to L1 after 2 L2 hits
        self.l2_access_counts: Dict[str, int] = defaultdict(int)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Track L2 access for promotion
            self.l2_access_counts[key] += 1
            
            # Promote to L1 if accessed frequently
            if self.l2_access_counts[key] >= self.promotion_threshold:
                self.l1_cache.put(key, value)
                del self.l2_access_counts[key]
            
            return value
        
        return None
    
    async def put(self, key: str, value: Any, level: int = 2) -> None:
        """Put value in cache at specified level."""
        if level == 1:
            self.l1_cache.put(key, value)
            # Also put in L2 for persistence
            self.l2_cache.put(key, value)
        else:
            self.l2_cache.put(key, value)
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels."""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats()
        }


class ResourcePool:
    """Generic resource pool for expensive objects."""
    
    def __init__(self, 
                 factory: Callable[[], T], 
                 max_size: int = 10,
                 cleanup: Optional[Callable[[T], None]] = None):
        """Initialize resource pool."""
        self.factory = factory
        self.cleanup = cleanup
        self.max_size = max_size
        self.pool: deque[T] = deque()
        self.created_count = 0
        self.borrowed_count = 0
        self.returned_count = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> T:
        """Acquire resource from pool."""
        async with self._lock:
            if self.pool:
                resource = self.pool.popleft()
            else:
                resource = self.factory()
                self.created_count += 1
            
            self.borrowed_count += 1
            return resource
    
    async def release(self, resource: T) -> None:
        """Release resource back to pool."""
        async with self._lock:
            if len(self.pool) < self.max_size:
                self.pool.append(resource)
                self.returned_count += 1
            else:
                # Pool full, cleanup resource
                if self.cleanup:
                    self.cleanup(resource)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            'available': len(self.pool),
            'created': self.created_count,
            'borrowed': self.borrowed_count,
            'returned': self.returned_count,
            'max_size': self.max_size
        }


class AsyncBatchProcessor:
    """Batch processing system for improved throughput."""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 max_concurrency: int = 5):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrency = max_concurrency
        self.pending_items: List[Dict[str, Any]] = []
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._last_batch_time = time.time()
    
    async def submit(self, item: Any, callback: Callable[[Any], Any]) -> Any:
        """Submit item for batch processing."""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_items.append({
                'item': item,
                'callback': callback,
                'future': future
            })
            
            # Process batch if conditions met
            should_process = (
                len(self.pending_items) >= self.batch_size or
                time.time() - self._last_batch_time >= self.max_wait_time
            )
            
            if should_process:
                await self._process_batch()
        
        return await future
    
    async def _process_batch(self) -> None:
        """Process current batch."""
        if not self.pending_items:
            return
        
        batch = self.pending_items.copy()
        self.pending_items.clear()
        self._last_batch_time = time.time()
        
        # Process batch with concurrency control
        async with self.semaphore:
            await self._execute_batch(batch)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Execute batch of items."""
        tasks = []
        
        for item_data in batch:
            task = asyncio.create_task(
                self._process_item(item_data)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_item(self, item_data: Dict[str, Any]) -> None:
        """Process individual item."""
        try:
            result = await item_data['callback'](item_data['item'])
            item_data['future'].set_result(result)
        except Exception as e:
            item_data['future'].set_exception(e)


class PerformanceMonitor:
    """Performance monitoring and profiling system."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize performance monitor."""
        self.history_size = history_size
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=history_size)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.alerts: List[str] = []
        self._lock = threading.Lock()
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self._lock:
            self.metrics_history.append(metrics)
            self.operation_stats[metrics.operation_name].append(metrics.execution_time)
            
            # Keep operation stats bounded
            if len(self.operation_stats[metrics.operation_name]) > 100:
                self.operation_stats[metrics.operation_name] = \
                    self.operation_stats[metrics.operation_name][-100:]
            
            # Check for performance alerts
            self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts."""
        # High latency alert
        if metrics.execution_time > 5.0:  # 5 seconds
            self.alerts.append(f"High latency detected: {metrics.operation_name} took {metrics.execution_time:.2f}s")
        
        # Low cache hit rate alert
        if metrics.cache_hit_rate < 0.5:  # Below 50%
            self.alerts.append(f"Low cache hit rate: {metrics.operation_name} has {metrics.cache_hit_rate:.1%} hit rate")
        
        # Keep alerts bounded
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            
            # Calculate percentiles
            execution_times = [m.execution_time for m in recent_metrics]
            execution_times.sort()
            
            p50 = execution_times[len(execution_times) // 2] if execution_times else 0
            p95 = execution_times[int(len(execution_times) * 0.95)] if execution_times else 0
            
            return {
                "avg_execution_time": avg_execution_time,
                "avg_cache_hit_rate": avg_cache_hit_rate,
                "avg_throughput": avg_throughput,
                "latency_p50": p50,
                "latency_p95": p95,
                "total_operations": len(self.metrics_history),
                "recent_alerts": self.alerts[-5:],  # Last 5 alerts
                "operation_breakdown": {
                    op: {
                        "count": len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
                    for op, times in self.operation_stats.items()
                }
            }


class AutoScaler:
    """Auto-scaling system based on performance metrics."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = 10,
                 target_latency: float = 1.0,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        """Initialize auto-scaler."""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_latency = target_latency
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_workers = min_workers
        self.last_scale_time = time.time()
        self.cooldown_period = 60  # 1 minute cooldown
        self.scaling_history: List[Dict[str, Any]] = []
        
    def should_scale(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Determine if scaling is needed based on metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.cooldown_period:
            return None
        
        # Scale up if latency is high
        if (metrics.execution_time > self.target_latency * self.scale_up_threshold and
            self.current_workers < self.max_workers):
            return "scale_up"
        
        # Scale down if latency is low and throughput is low
        if (metrics.execution_time < self.target_latency * self.scale_down_threshold and
            metrics.throughput < 10 and  # Low throughput
            self.current_workers > self.min_workers):
            return "scale_down"
        
        return None
    
    def execute_scaling(self, action: str, reason: str) -> Dict[str, Any]:
        """Execute scaling action."""
        old_workers = self.current_workers
        
        if action == "scale_up":
            self.current_workers = min(self.current_workers + 1, self.max_workers)
        elif action == "scale_down":
            self.current_workers = max(self.current_workers - 1, self.min_workers)
        
        self.last_scale_time = time.time()
        
        scaling_event = {
            "timestamp": self.last_scale_time,
            "action": action,
            "reason": reason,
            "old_workers": old_workers,
            "new_workers": self.current_workers
        }
        
        self.scaling_history.append(scaling_event)
        
        # Keep history bounded
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
        
        logger.info(f"Auto-scaling: {action} from {old_workers} to {self.current_workers} workers. Reason: {reason}")
        
        return scaling_event


class PerformanceOptimizer:
    """Main performance optimization framework."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache = MultiLevelCache()
        self.monitor = PerformanceMonitor()
        self.auto_scaler = AutoScaler()
        self.batch_processor = AsyncBatchProcessor()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
        
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached(self, ttl: Optional[float] = None, level: int = 2):
        """Decorator for caching function results."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                start_time = time.time()
                cached_result = await self.cache.get(cache_key)
                
                if cached_result is not None:
                    # Cache hit
                    execution_time = time.time() - start_time
                    
                    metrics = PerformanceMetrics(
                        operation_name=func.__name__,
                        execution_time=execution_time,
                        memory_usage=0,  # Cached result has minimal memory impact
                        cpu_usage=0.1,  # Minimal CPU for cache lookup
                        cache_hit_rate=1.0,  # This was a cache hit
                        throughput=1.0 / execution_time if execution_time > 0 else float('inf'),
                        latency_p95=execution_time
                    )
                    
                    self.monitor.record_metrics(metrics)
                    return cached_result
                
                # Cache miss - execute function
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Store in cache
                await self.cache.put(cache_key, result, level=level)
                
                # Record metrics
                metrics = PerformanceMetrics(
                    operation_name=func.__name__,
                    execution_time=execution_time,
                    memory_usage=self._estimate_memory_usage(result),
                    cpu_usage=0.8,  # Assume high CPU for actual computation
                    cache_hit_rate=0.0,  # This was a cache miss
                    throughput=1.0 / execution_time if execution_time > 0 else float('inf'),
                    latency_p95=execution_time
                )
                
                self.monitor.record_metrics(metrics)
                
                # Check for auto-scaling
                scaling_action = self.auto_scaler.should_scale(metrics)
                if scaling_action:
                    self.auto_scaler.execute_scaling(
                        scaling_action, 
                        f"Latency: {execution_time:.2f}s, Target: {self.auto_scaler.target_latency:.2f}s"
                    )
                
                return result
            
            return wrapper
        return decorator
    
    def batched(self, batch_size: int = 10, max_wait_time: float = 1.0):
        """Decorator for batch processing."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.batch_processor.submit(
                    (args, kwargs),
                    lambda item: func(*item[0], **item[1])
                )
            
            return wrapper
        return decorator
    
    def pooled_resource(self, resource_name: str, factory: Callable[[], T], max_size: int = 10):
        """Context manager for pooled resources."""
        if resource_name not in self.resource_pools:
            self.resource_pools[resource_name] = ResourcePool(factory, max_size)
        
        class PooledResourceContext:
            def __init__(self, pool: ResourcePool):
                self.pool = pool
                self.resource = None
            
            async def __aenter__(self):
                self.resource = await self.pool.acquire()
                return self.resource
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.resource:
                    await self.pool.release(self.resource)
        
        return PooledResourceContext(self.resource_pools[resource_name])
    
    def _estimate_memory_usage(self, obj: Any) -> int:
        """Estimate memory usage of an object."""
        import sys
        try:
            return sys.getsizeof(obj)
        except:
            return 1024  # Default estimate
    
    async def optimize_operation(self, operation_name: str, operation: Callable) -> Any:
        """Optimize a generic operation with all available techniques."""
        start_time = time.time()
        
        try:
            # Execute with monitoring
            result = await operation()
            
            execution_time = time.time() - start_time
            
            # Record comprehensive metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage=self._estimate_memory_usage(result),
                cpu_usage=0.7,  # Estimated CPU usage
                cache_hit_rate=0.5,  # Estimated cache hit rate
                throughput=1.0 / execution_time if execution_time > 0 else float('inf'),
                latency_p95=execution_time
            )
            
            self.monitor.record_metrics(metrics)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure metrics
            metrics = PerformanceMetrics(
                operation_name=f"{operation_name}_failed",
                execution_time=execution_time,
                memory_usage=0,
                cpu_usage=0.1,
                cache_hit_rate=0.0,
                throughput=0.0,
                latency_p95=execution_time
            )
            
            self.monitor.record_metrics(metrics)
            raise e
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        cache_stats = self.cache.get_stats()
        performance_summary = self.monitor.get_performance_summary()
        
        pool_stats = {
            name: pool.get_stats() 
            for name, pool in self.resource_pools.items()
        }
        
        return {
            "cache_performance": cache_stats,
            "execution_performance": performance_summary,
            "resource_pools": pool_stats,
            "auto_scaling": {
                "current_workers": self.auto_scaler.current_workers,
                "recent_scaling_events": self.auto_scaler.scaling_history[-5:],
                "scaling_config": {
                    "min_workers": self.auto_scaler.min_workers,
                    "max_workers": self.auto_scaler.max_workers,
                    "target_latency": self.auto_scaler.target_latency
                }
            },
            "batch_processing": {
                "pending_items": len(self.batch_processor.pending_items),
                "max_concurrency": self.batch_processor.max_concurrency,
                "batch_size": self.batch_processor.batch_size
            },
            "optimization_recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        cache_stats = self.cache.get_stats()
        performance_summary = self.monitor.get_performance_summary()
        
        # Cache recommendations
        if isinstance(cache_stats, dict):
            l1_stats = cache_stats.get('l1')
            if l1_stats and l1_stats.hit_rate < 0.7:
                recommendations.append("Consider increasing L1 cache size or TTL - low hit rate detected")
        
        # Performance recommendations
        if isinstance(performance_summary, dict):
            avg_latency = performance_summary.get('avg_execution_time', 0)
            if avg_latency > 2.0:
                recommendations.append("High average latency detected - consider optimizing critical paths")
            
            p95_latency = performance_summary.get('latency_p95', 0)
            if p95_latency > 5.0:
                recommendations.append("High P95 latency - investigate tail latency issues")
        
        # Auto-scaling recommendations
        if self.auto_scaler.current_workers == self.auto_scaler.max_workers:
            recommendations.append("Auto-scaler at maximum capacity - consider increasing max_workers")
        
        if not recommendations:
            recommendations.append("System performance appears optimal")
        
        return recommendations