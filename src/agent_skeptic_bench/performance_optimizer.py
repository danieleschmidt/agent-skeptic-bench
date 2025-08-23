"""Performance Optimization System for Agent Skeptic Bench

Provides advanced caching, concurrent processing, resource pooling,
load balancing, and auto-scaling for maximum performance and scalability.
"""

import asyncio
import hashlib
import json
import logging
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import weakref
from functools import wraps, lru_cache
import statistics

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def access(self) -> Any:
        """Access cache entry and update metadata."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
        return self.value


@dataclass
class WorkerStats:
    """Worker performance statistics."""
    worker_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_response_time_ms: float = 0.0
    current_connections: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    weight: float = 1.0
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate."""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks
    
    @property
    def load_score(self) -> float:
        """Calculate load score for balancing."""
        # Lower is better
        return (self.current_connections + self.avg_response_time_ms / 1000.0) / max(self.weight, 0.1)


class AdaptiveCache:
    """High-performance adaptive cache with multiple strategies."""
    
    def __init__(self, 
                 max_size: int = 10000,
                 default_ttl: int = 3600,  # 1 hour
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 cleanup_interval: int = 300):  # 5 minutes
        """Initialize adaptive cache."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cleanup_interval = cleanup_interval
        
        # Storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_bytes": 0,
            "avg_access_time_ms": 0.0
        }
        
        # Cleanup
        self.last_cleanup = datetime.utcnow()
        self._cache_lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._cache_lock:
            start_time = time.time()
            
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None
            
            # Update access patterns
            value = entry.access()
            self._update_access_pattern(key)
            
            self.stats["hits"] += 1
            access_time = (time.time() - start_time) * 1000
            self._update_avg_access_time(access_time)
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._cache_lock:
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))
            
            # Check if we need to make space
            if len(self.cache) >= self.max_size and key not in self.cache:
                if not self._evict_entries(1):
                    return False  # Could not make space
            
            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats["size_bytes"] -= old_entry.size_bytes
            
            self.cache[key] = entry
            self.stats["size_bytes"] += size_bytes
            self._update_access_pattern(key)
            
            # Cleanup if needed
            if datetime.utcnow() - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
                self._cleanup_expired()
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._cache_lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.stats["size_bytes"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "size_bytes": self.stats["size_bytes"],
                "hit_rate": hit_rate,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "avg_access_time_ms": self.stats["avg_access_time_ms"],
                "strategy": self.strategy.value
            }
    
    def _update_access_pattern(self, key: str):
        """Update access patterns for different strategies."""
        if self.strategy == CacheStrategy.LRU:
            # Move to end of access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        elif self.strategy == CacheStrategy.LFU:
            self.frequency_counter[key] += 1
    
    def _evict_entries(self, count: int) -> bool:
        """Evict entries based on strategy."""
        evicted = 0
        
        if self.strategy == CacheStrategy.LRU:
            while evicted < count and self.access_order:
                key = self.access_order.popleft()
                if key in self.cache:
                    self._remove_entry(key)
                    evicted += 1
                    
        elif self.strategy == CacheStrategy.LFU:
            # Sort by frequency
            entries = [(self.frequency_counter[k], k) for k in self.cache.keys()]
            entries.sort()
            
            for _, key in entries[:count]:
                self._remove_entry(key)
                evicted += 1
                
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k].created_at)
            
            for key in oldest_keys[:count]:
                self._remove_entry(key)
                evicted += 1
        
        self.stats["evictions"] += evicted
        return evicted == count
    
    def _remove_entry(self, key: str):
        """Remove entry and update statistics."""
        if key in self.cache:
            entry = self.cache[key]
            self.stats["size_bytes"] -= entry.size_bytes
            del self.cache[key]
        
        if key in self.access_order:
            self.access_order.remove(key)
        
        if key in self.frequency_counter:
            del self.frequency_counter[key]
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        self.last_cleanup = datetime.utcnow()
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        if self.stats["avg_access_time_ms"] == 0:
            self.stats["avg_access_time_ms"] = access_time_ms
        else:
            self.stats["avg_access_time_ms"] = (
                alpha * access_time_ms + 
                (1 - alpha) * self.stats["avg_access_time_ms"]
            )


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED):
        """Initialize load balancer."""
        self.strategy = strategy
        self.workers: Dict[str, WorkerStats] = {}
        self.round_robin_index = 0
        self.balancer_lock = threading.RLock()
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_rebalance = datetime.utcnow()
        self.rebalance_interval = 30  # seconds
    
    def register_worker(self, worker_id: str, weight: float = 1.0):
        """Register a new worker."""
        with self.balancer_lock:
            self.workers[worker_id] = WorkerStats(
                worker_id=worker_id,
                weight=weight
            )
            logger.info(f"Registered worker: {worker_id} with weight {weight}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        with self.balancer_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                if worker_id in self.response_times:
                    del self.response_times[worker_id]
                logger.info(f"Unregistered worker: {worker_id}")
    
    def select_worker(self) -> Optional[str]:
        """Select optimal worker based on strategy."""
        with self.balancer_lock:
            if not self.workers:
                return None
            
            # Update worker statistics if needed
            self._update_worker_stats()
            
            active_workers = [
                w for w in self.workers.values()
                if (datetime.utcnow() - w.last_activity).total_seconds() < 300  # 5 minutes
            ]
            
            if not active_workers:
                active_workers = list(self.workers.values())
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                worker = active_workers[self.round_robin_index % len(active_workers)]
                self.round_robin_index += 1
                return worker.worker_id
                
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                worker = min(active_workers, key=lambda w: w.current_connections)
                return worker.worker_id
                
            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                # Weighted random selection
                weights = [w.weight for w in active_workers]
                total_weight = sum(weights)
                
                if total_weight == 0:
                    return active_workers[0].worker_id
                
                import random
                r = random.uniform(0, total_weight)
                cumulative = 0
                
                for worker, weight in zip(active_workers, weights):
                    cumulative += weight
                    if r <= cumulative:
                        return worker.worker_id
                
                return active_workers[-1].worker_id
                
            elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                # Select worker with lowest load score
                worker = min(active_workers, key=lambda w: w.load_score)
                return worker.worker_id
            
            return active_workers[0].worker_id if active_workers else None
    
    def record_task_start(self, worker_id: str):
        """Record task start for a worker."""
        with self.balancer_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_connections += 1
                worker.total_tasks += 1
                worker.last_activity = datetime.utcnow()
    
    def record_task_completion(self, worker_id: str, response_time_ms: float, success: bool = True):
        """Record task completion for a worker."""
        with self.balancer_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.current_connections = max(0, worker.current_connections - 1)
                
                if success:
                    worker.completed_tasks += 1
                else:
                    worker.failed_tasks += 1
                
                # Update response time
                self.response_times[worker_id].append(response_time_ms)
                worker.last_activity = datetime.utcnow()
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers."""
        with self.balancer_lock:
            self._update_worker_stats()
            
            return {
                worker_id: {
                    "total_tasks": worker.total_tasks,
                    "completed_tasks": worker.completed_tasks,
                    "failed_tasks": worker.failed_tasks,
                    "success_rate": worker.success_rate,
                    "avg_response_time_ms": worker.avg_response_time_ms,
                    "current_connections": worker.current_connections,
                    "load_score": worker.load_score,
                    "weight": worker.weight,
                    "last_activity": worker.last_activity.isoformat()
                }
                for worker_id, worker in self.workers.items()
            }
    
    def _update_worker_stats(self):
        """Update worker performance statistics."""
        for worker_id, worker in self.workers.items():
            if worker_id in self.response_times:
                times = list(self.response_times[worker_id])
                if times:
                    worker.avg_response_time_ms = statistics.mean(times)


class ResourcePool:
    """Generic resource pool for connection pooling and resource management."""
    
    def __init__(self, 
                 resource_factory: Callable[[], Any],
                 min_size: int = 5,
                 max_size: int = 50,
                 idle_timeout: int = 300):  # 5 minutes
        """Initialize resource pool."""
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        # Pool management
        self.available_resources: deque = deque()
        self.in_use_resources: Dict[Any, datetime] = {}
        self.resource_stats: Dict[Any, Dict[str, Any]] = {}
        self.pool_lock = threading.RLock()
        
        # Initialize minimum resources
        self._ensure_min_resources()
        
        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_active = True
        self.cleanup_thread.start()
    
    def acquire(self, timeout: Optional[int] = None) -> Optional[Any]:
        """Acquire a resource from the pool."""
        start_time = time.time()
        
        with self.pool_lock:
            # Try to get available resource
            if self.available_resources:
                resource = self.available_resources.popleft()
                self.in_use_resources[resource] = datetime.utcnow()
                return resource
            
            # Create new resource if under limit
            if len(self.in_use_resources) < self.max_size:
                try:
                    resource = self.resource_factory()
                    self.in_use_resources[resource] = datetime.utcnow()
                    self.resource_stats[resource] = {
                        "created_at": datetime.utcnow(),
                        "use_count": 0,
                        "total_time_ms": 0.0
                    }
                    return resource
                except Exception as e:
                    logger.error(f"Failed to create resource: {e}")
                    return None
        
        # Wait for resource to become available
        if timeout:
            end_time = start_time + timeout
            while time.time() < end_time:
                time.sleep(0.1)
                with self.pool_lock:
                    if self.available_resources:
                        resource = self.available_resources.popleft()
                        self.in_use_resources[resource] = datetime.utcnow()
                        return resource
        
        return None
    
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        with self.pool_lock:
            if resource in self.in_use_resources:
                acquired_at = self.in_use_resources[resource]
                del self.in_use_resources[resource]
                
                # Update statistics
                if resource in self.resource_stats:
                    stats = self.resource_stats[resource]
                    stats["use_count"] += 1
                    stats["total_time_ms"] += (datetime.utcnow() - acquired_at).total_seconds() * 1000
                
                # Return to available pool
                self.available_resources.append(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self.pool_lock:
            total_resources = len(self.available_resources) + len(self.in_use_resources)
            
            return {
                "total_resources": total_resources,
                "available_resources": len(self.available_resources),
                "in_use_resources": len(self.in_use_resources),
                "min_size": self.min_size,
                "max_size": self.max_size,
                "utilization": len(self.in_use_resources) / self.max_size if self.max_size > 0 else 0,
                "avg_use_count": statistics.mean([
                    stats["use_count"] for stats in self.resource_stats.values()
                ]) if self.resource_stats else 0
            }
    
    def _ensure_min_resources(self):
        """Ensure minimum number of resources are available."""
        with self.pool_lock:
            total_resources = len(self.available_resources) + len(self.in_use_resources)
            
            while total_resources < self.min_size:
                try:
                    resource = self.resource_factory()
                    self.available_resources.append(resource)
                    self.resource_stats[resource] = {
                        "created_at": datetime.utcnow(),
                        "use_count": 0,
                        "total_time_ms": 0.0
                    }
                    total_resources += 1
                except Exception as e:
                    logger.error(f"Failed to create minimum resource: {e}")
                    break
    
    def _cleanup_loop(self):
        """Background cleanup of idle resources."""
        while self.cleanup_active:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception as e:
                logger.error(f"Resource pool cleanup error: {e}")
    
    def _cleanup_idle_resources(self):
        """Clean up idle resources beyond minimum."""
        with self.pool_lock:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.idle_timeout)
            
            # Remove idle resources beyond minimum
            total_resources = len(self.available_resources) + len(self.in_use_resources)
            
            while len(self.available_resources) > 0 and total_resources > self.min_size:
                resource = self.available_resources[0]
                
                # Check if resource has been idle too long
                if resource in self.resource_stats:
                    created_at = self.resource_stats[resource]["created_at"]
                    if created_at < cutoff_time:
                        self.available_resources.popleft()
                        del self.resource_stats[resource]
                        total_resources -= 1
                        logger.debug("Cleaned up idle resource")
                        continue
                
                break


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache = AdaptiveCache(max_size=50000, default_ttl=3600)
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.PERFORMANCE_BASED)
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        
        # Performance monitoring
        self.performance_metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_response_time_ms": 0.0,
            "peak_concurrent_requests": 0,
            "current_concurrent_requests": 0,
            "total_processing_time_ms": 0.0
        }
        
        self.metrics_lock = threading.RLock()
        
    def register_resource_pool(self, name: str, factory: Callable[[], Any],
                             min_size: int = 5, max_size: int = 50):
        """Register a resource pool."""
        self.resource_pools[name] = ResourcePool(
            resource_factory=factory,
            min_size=min_size,
            max_size=max_size
        )
        logger.info(f"Registered resource pool: {name}")
    
    def get_cached_or_compute(self, key: str, compute_func: Callable[[], Any],
                            ttl: Optional[int] = None) -> Any:
        """Get value from cache or compute and cache it."""
        # Try cache first
        cached_value = self.cache.get(key)
        if cached_value is not None:
            with self.metrics_lock:
                self.performance_metrics["cache_hits"] += 1
            return cached_value
        
        # Compute and cache
        start_time = time.time()
        value = compute_func()
        compute_time = (time.time() - start_time) * 1000
        
        # Cache the result
        self.cache.set(key, value, ttl)
        
        # Update metrics
        with self.metrics_lock:
            self.performance_metrics["total_processing_time_ms"] += compute_time
        
        return value
    
    async def process_concurrent_tasks(self, tasks: List[Callable], 
                                     max_concurrency: int = 10) -> List[Any]:
        """Process tasks with optimal concurrency."""
        with self.metrics_lock:
            self.performance_metrics["current_concurrent_requests"] += len(tasks)
            self.performance_metrics["peak_concurrent_requests"] = max(
                self.performance_metrics["peak_concurrent_requests"],
                self.performance_metrics["current_concurrent_requests"]
            )
        
        try:
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrency)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(task):
                            result = await task()
                        else:
                            result = await asyncio.get_event_loop().run_in_executor(
                                self.thread_pool, task
                            )
                        
                        response_time = (time.time() - start_time) * 1000
                        self._update_response_time(response_time)
                        
                        return result
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        return {"error": str(e)}
            
            # Execute all tasks concurrently
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )
            
            return results
            
        finally:
            with self.metrics_lock:
                self.performance_metrics["current_concurrent_requests"] -= len(tasks)
                self.performance_metrics["total_requests"] += len(tasks)
    
    def get_optimized_worker(self) -> Optional[str]:
        """Get optimally balanced worker."""
        return self.load_balancer.select_worker()
    
    def record_worker_performance(self, worker_id: str, response_time_ms: float, success: bool = True):
        """Record worker performance metrics."""
        self.load_balancer.record_task_completion(worker_id, response_time_ms, success)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        with self.metrics_lock:
            total_requests = self.performance_metrics["total_requests"]
            
            dashboard = {
                "performance_summary": {
                    "total_requests": total_requests,
                    "cache_hit_rate": self.performance_metrics["cache_hits"] / total_requests if total_requests > 0 else 0,
                    "avg_response_time_ms": self.performance_metrics["avg_response_time_ms"],
                    "peak_concurrent_requests": self.performance_metrics["peak_concurrent_requests"],
                    "current_concurrent_requests": self.performance_metrics["current_concurrent_requests"]
                },
                "cache_stats": self.cache.get_stats(),
                "load_balancer_stats": self.load_balancer.get_worker_stats(),
                "resource_pools": {
                    name: pool.get_stats() 
                    for name, pool in self.resource_pools.items()
                },
                "thread_pool_stats": {
                    "max_workers": self.thread_pool._max_workers,
                    "active_threads": len(self.thread_pool._threads) if hasattr(self.thread_pool, '_threads') else 0
                }
            }
            
            return dashboard
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """Analyze and optimize system performance."""
        dashboard = self.get_performance_dashboard()
        recommendations = []
        
        # Cache optimization
        cache_hit_rate = dashboard["performance_summary"]["cache_hit_rate"]
        if cache_hit_rate < 0.7:
            recommendations.append("Consider increasing cache size or TTL for better hit rates")
        
        # Concurrency optimization
        peak_concurrent = dashboard["performance_summary"]["peak_concurrent_requests"]
        current_concurrent = dashboard["performance_summary"]["current_concurrent_requests"]
        
        if peak_concurrent > 50:
            recommendations.append("Consider implementing auto-scaling for high load scenarios")
        
        # Resource pool optimization
        for pool_name, pool_stats in dashboard["resource_pools"].items():
            utilization = pool_stats["utilization"]
            if utilization > 0.8:
                recommendations.append(f"Resource pool '{pool_name}' is highly utilized - consider increasing max_size")
            elif utilization < 0.2:
                recommendations.append(f"Resource pool '{pool_name}' is underutilized - consider reducing min_size")
        
        # Load balancing optimization
        worker_stats = dashboard["load_balancer_stats"]
        if worker_stats:
            success_rates = [stats["success_rate"] for stats in worker_stats.values()]
            min_success_rate = min(success_rates) if success_rates else 1.0
            
            if min_success_rate < 0.9:
                recommendations.append("Some workers have low success rates - investigate worker health")
        
        return {
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "performance_dashboard": dashboard,
            "recommendations": recommendations,
            "optimizations_applied": self._apply_automatic_optimizations(dashboard)
        }
    
    def _apply_automatic_optimizations(self, dashboard: Dict[str, Any]) -> List[str]:
        """Apply automatic performance optimizations."""
        optimizations = []
        
        # Auto-adjust cache size
        cache_stats = dashboard["cache_stats"]
        hit_rate = cache_stats["hit_rate"]
        
        if hit_rate > 0.9 and cache_stats["entries"] > cache_stats["max_size"] * 0.9:
            new_size = int(cache_stats["max_size"] * 1.2)  # Increase by 20%
            self.cache.max_size = min(new_size, 100000)  # Cap at 100k
            optimizations.append(f"Increased cache size to {self.cache.max_size}")
        
        # Auto-register workers based on load
        current_concurrent = dashboard["performance_summary"]["current_concurrent_requests"]
        if current_concurrent > 30 and len(self.load_balancer.workers) < 10:
            # This would trigger external worker provisioning in a real system
            optimizations.append("Recommended scaling up workers due to high load")
        
        return optimizations
    
    def _update_response_time(self, response_time_ms: float):
        """Update average response time with exponential smoothing."""
        with self.metrics_lock:
            alpha = 0.1  # Smoothing factor
            if self.performance_metrics["avg_response_time_ms"] == 0:
                self.performance_metrics["avg_response_time_ms"] = response_time_ms
            else:
                self.performance_metrics["avg_response_time_ms"] = (
                    alpha * response_time_ms + 
                    (1 - alpha) * self.performance_metrics["avg_response_time_ms"]
                )


# Global optimizer instance
_global_optimizer = None

def get_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()}"
            
            # Get from cache or compute
            def compute():
                if asyncio.iscoroutinefunction(func):
                    return asyncio.run(func(*args, **kwargs))
                return func(*args, **kwargs)
            
            return optimizer.get_cached_or_compute(cache_key, compute, ttl)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()}"
            
            # Get from cache or compute
            def compute():
                return func(*args, **kwargs)
            
            return optimizer.get_cached_or_compute(cache_key, compute, ttl)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator