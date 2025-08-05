"""Performance optimization utilities for Agent Skeptic Bench."""

import asyncio
import functools
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

from .models import Scenario, SkepticResponse, EvaluationResult
from .exceptions import AgentTimeoutError


T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking system performance."""
    
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    active_connections: int
    evaluation_queue_size: int
    cache_hit_rate: float
    average_response_time_ms: float
    requests_per_second: float
    concurrent_evaluations: int


class PerformanceCache:
    """High-performance in-memory cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self._misses += 1
            return None
        
        value, expiry_time = self._cache[key]
        current_time = time.time()
        
        if current_time > expiry_time:
            # Expired, remove from cache
            del self._cache[key]
            del self._access_times[key]
            self._misses += 1
            return None
        
        # Update access time for LRU
        self._access_times[key] = current_time
        self._hits += 1
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl
        self._cache[key] = (value, expiry_time)
        self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "utilization": len(self._cache) / self.max_size
        }


class ConcurrentEvaluator:
    """High-performance concurrent evaluation manager."""
    
    def __init__(self, 
                 max_concurrent: int = None,
                 timeout_seconds: float = 60.0,
                 use_process_pool: bool = False):
        self.max_concurrent = max_concurrent or min(32, (psutil.cpu_count() or 1) * 4)
        self.timeout_seconds = timeout_seconds
        self.use_process_pool = use_process_pool
        
        # Initialize pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent)
        if use_process_pool:
            process_workers = min(psutil.cpu_count() or 1, 8)
            self.process_pool = ProcessPoolExecutor(max_workers=process_workers)
        else:
            self.process_pool = None
        
        # Semaphores for rate limiting
        self.evaluation_semaphore = asyncio.Semaphore(self.max_concurrent)
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent // 2)  # More conservative for API calls
        
        # Performance tracking
        self.active_evaluations = 0
        self.completed_evaluations = 0
        self.failed_evaluations = 0
        self.total_evaluation_time = 0.0
    
    async def evaluate_concurrent(self, 
                                evaluation_tasks: List[Callable],
                                batch_size: int = None) -> List[Any]:
        """Execute evaluation tasks concurrently with optimal batching."""
        batch_size = batch_size or min(self.max_concurrent, len(evaluation_tasks))
        results = []
        
        # Process tasks in batches to avoid overwhelming the system
        for i in range(0, len(evaluation_tasks), batch_size):
            batch = evaluation_tasks[i:i + batch_size]
            batch_results = await self._execute_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _execute_batch(self, tasks: List[Callable]) -> List[Any]:
        """Execute a batch of tasks concurrently."""
        async def execute_with_semaphore(task):
            async with self.evaluation_semaphore:
                self.active_evaluations += 1
                start_time = time.time()
                
                try:
                    result = await asyncio.wait_for(task(), timeout=self.timeout_seconds)
                    self.completed_evaluations += 1
                    return result
                except asyncio.TimeoutError:
                    self.failed_evaluations += 1
                    raise AgentTimeoutError("batch_evaluator", self.timeout_seconds)
                except Exception as e:
                    self.failed_evaluations += 1
                    raise
                finally:
                    self.active_evaluations -= 1
                    self.total_evaluation_time += time.time() - start_time
        
        return await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
    
    async def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
    
    @property
    def performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_evals = self.completed_evaluations + self.failed_evaluations
        avg_time = self.total_evaluation_time / total_evals if total_evals > 0 else 0
        
        return {
            "active_evaluations": self.active_evaluations,
            "completed_evaluations": self.completed_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "success_rate": self.completed_evaluations / total_evals if total_evals > 0 else 0,
            "average_evaluation_time_s": avg_time,
            "evaluations_per_second": total_evals / self.total_evaluation_time if self.total_evaluation_time > 0 else 0
        }


class PerformanceOptimizer:
    """Adaptive performance optimizer that adjusts system parameters."""
    
    def __init__(self):
        self.cache = PerformanceCache()
        self.evaluator = ConcurrentEvaluator()
        self._monitoring_active = False
        self._optimization_history = []
    
    def cache_scenario_metrics(self, scenario_id: str, response_hash: str, 
                             metrics: Dict[str, float], ttl: int = 3600):
        """Cache scenario evaluation metrics."""
        cache_key = f"metrics_{scenario_id}_{response_hash}"
        self.cache.set(cache_key, metrics, ttl)
    
    def get_cached_metrics(self, scenario_id: str, response_hash: str) -> Optional[Dict[str, float]]:
        """Get cached metrics for scenario/response combination."""
        cache_key = f"metrics_{scenario_id}_{response_hash}"
        return self.cache.get(cache_key)
    
    def cache_scenario_analysis(self, scenario_id: str, analysis: Dict[str, Any], ttl: int = 7200):
        """Cache scenario analysis results."""
        cache_key = f"analysis_{scenario_id}"
        self.cache.set(cache_key, analysis, ttl)
    
    def get_cached_analysis(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis for scenario."""
        cache_key = f"analysis_{scenario_id}"
        return self.cache.get(cache_key)
    
    async def optimize_concurrent_evaluation(self, 
                                           evaluation_tasks: List[Callable],
                                           target_latency_ms: float = 2000) -> List[Any]:
        """Optimize concurrent evaluation based on system performance."""
        # Monitor current system performance
        metrics = self._get_system_metrics()
        
        # Adjust concurrency based on system load
        optimal_concurrency = self._calculate_optimal_concurrency(metrics, target_latency_ms)
        self.evaluator.max_concurrent = optimal_concurrency
        
        # Execute with optimized settings
        results = await self.evaluator.evaluate_concurrent(evaluation_tasks)
        
        # Record optimization results
        self._record_optimization(metrics, optimal_concurrency, len(evaluation_tasks))
        
        return results
    
    def _get_system_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_connections=len(psutil.net_connections()),
            evaluation_queue_size=self.evaluator.active_evaluations,
            cache_hit_rate=self.cache.hit_rate,
            average_response_time_ms=0.0,  # Will be calculated
            requests_per_second=0.0,  # Will be calculated
            concurrent_evaluations=self.evaluator.active_evaluations
        )
    
    def _calculate_optimal_concurrency(self, 
                                     metrics: PerformanceMetrics,
                                     target_latency_ms: float) -> int:
        """Calculate optimal concurrency based on system metrics."""
        base_concurrency = psutil.cpu_count() or 1
        
        # Adjust based on CPU usage
        if metrics.cpu_usage_percent > 80:
            cpu_factor = 0.5
        elif metrics.cpu_usage_percent > 60:
            cpu_factor = 0.7
        else:
            cpu_factor = 1.0
        
        # Adjust based on memory usage
        memory_usage_percent = (metrics.memory_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if memory_usage_percent > 80:
            memory_factor = 0.6
        elif memory_usage_percent > 60:
            memory_factor = 0.8
        else:
            memory_factor = 1.0
        
        # Calculate optimal concurrency
        optimal = int(base_concurrency * 4 * cpu_factor * memory_factor)
        
        # Ensure reasonable bounds
        return max(1, min(optimal, 64))
    
    def _record_optimization(self, metrics: PerformanceMetrics, 
                           concurrency: int, num_tasks: int):
        """Record optimization decision for future analysis."""
        self._optimization_history.append({
            "timestamp": time.time(),
            "cpu_usage": metrics.cpu_usage_percent,
            "memory_usage_mb": metrics.memory_usage_mb,
            "concurrency": concurrency,
            "num_tasks": num_tasks,
            "cache_hit_rate": metrics.cache_hit_rate
        })
        
        # Keep only recent history
        if len(self._optimization_history) > 1000:
            self._optimization_history = self._optimization_history[-500:]
    
    async def shutdown(self):
        """Shutdown performance optimizer."""
        await self.evaluator.shutdown()
        self.cache.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "cache_stats": self.cache.stats,
            "evaluator_stats": self.evaluator.performance_stats,
            "optimization_history_length": len(self._optimization_history),
            "system_metrics": self._get_system_metrics().__dict__
        }


# Global performance optimizer instance
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def performance_cache(ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = PerformanceCache(default_ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = cache._generate_key(*args, **kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        # Add cache stats to wrapper
        wrapper.cache_stats = lambda: cache.stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


class AutoScalingManager:
    """Advanced auto-scaling manager for quantum-enhanced evaluation systems."""
    
    def __init__(self):
        self.scaling_history = []
        self.performance_thresholds = {
            "cpu_high": 75.0,
            "memory_high": 80.0,
            "response_time_high": 3000,  # ms
            "queue_depth_high": 100,
        }
        self.scaling_factors = {
            "scale_up_cpu": 1.5,
            "scale_up_memory": 1.3,
            "scale_up_latency": 2.0,
            "scale_down": 0.7
        }
        self.min_replicas = 1
        self.max_replicas = 20
        self.cooldown_period = 300  # seconds
        self.last_scaling_action = 0
    
    def analyze_scaling_need(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze current metrics and determine scaling requirements."""
        current_time = time.time()
        
        if current_time - self.last_scaling_action < self.cooldown_period:
            return {"action": "wait", "reason": "Cooldown period active"}
        
        scaling_signals = []
        
        # CPU-based scaling
        if metrics.cpu_usage_percent > self.performance_thresholds["cpu_high"]:
            scaling_signals.append({
                "metric": "cpu",
                "action": "scale_up",
                "factor": self.scaling_factors["scale_up_cpu"]
            })
        
        return self._determine_scaling_action(scaling_signals)
    
    def _determine_scaling_action(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the optimal scaling action from multiple signals."""
        if not signals:
            return {"action": "maintain", "reason": "All metrics within normal range"}
        
        scale_up_signals = [s for s in signals if s["action"] == "scale_up"]
        
        if scale_up_signals:
            most_urgent = scale_up_signals[0]
            return {
                "action": "scale_up",
                "scale_factor": most_urgent["factor"],
                "all_signals": signals
            }
        
        return {"action": "maintain", "reason": "Mixed signals, maintaining current scale"}
    
    def calculate_target_replicas(self, current_replicas: int, scaling_decision: Dict[str, Any]) -> int:
        """Calculate target number of replicas based on scaling decision."""
        if scaling_decision["action"] == "maintain":
            return current_replicas
        
        if scaling_decision["action"] == "scale_up":
            factor = scaling_decision["scale_factor"]
            target = int(current_replicas * factor)
        else:
            return current_replicas
        
        # Apply bounds
        target = max(self.min_replicas, min(target, self.max_replicas))
        
        # Record scaling decision
        self.scaling_history.append({
            "timestamp": time.time(),
            "current_replicas": current_replicas,
            "target_replicas": target,
            "action": scaling_decision["action"]
        })
        
        if scaling_decision["action"] in ["scale_up", "scale_down"]:
            self.last_scaling_action = time.time()
        
        return target


class LoadBalancer:
    """Intelligent load balancer for distributing evaluation tasks."""
    
    def __init__(self):
        self.workers = {}
        self.load_distribution_strategy = "quantum_weighted"
    
    def register_worker(self, worker_id: str, capacity: int, quantum_coherence: float = 1.0):
        """Register a new worker with quantum-enhanced capabilities."""
        self.workers[worker_id] = {
            "capacity": capacity,
            "current_load": 0,
            "quantum_coherence": quantum_coherence,
            "health_status": "healthy",
            "completed_tasks": 0,
            "failed_tasks": 0
        }
    
    def select_optimal_worker(self, task_complexity: float = 0.5) -> Optional[str]:
        """Select optimal worker using quantum-inspired load balancing."""
        healthy_workers = {
            wid: worker for wid, worker in self.workers.items()
            if worker["health_status"] == "healthy"
        }
        
        if not healthy_workers:
            return None
        
        if self.load_distribution_strategy == "quantum_weighted":
            return self._quantum_weighted_selection(healthy_workers, task_complexity)
        else:
            return list(healthy_workers.keys())[0]  # Fallback
    
    def _quantum_weighted_selection(self, workers: Dict[str, Dict], task_complexity: float) -> str:
        """Select worker using quantum-inspired weighting algorithm."""
        worker_scores = {}
        
        for worker_id, worker in workers.items():
            # Base capacity score
            capacity_score = (worker["capacity"] - worker["current_load"]) / worker["capacity"]
            
            # Quantum coherence bonus
            coherence_bonus = worker["quantum_coherence"] * 0.3
            
            # Performance history factor
            if worker["completed_tasks"] > 0:
                success_rate = worker["completed_tasks"] / (worker["completed_tasks"] + worker["failed_tasks"])
                performance_factor = success_rate * 0.4
            else:
                performance_factor = 0.2  # Neutral for new workers
            
            # Quantum superposition of scoring factors
            quantum_score = capacity_score * 0.4 + coherence_bonus + performance_factor
            
            worker_scores[worker_id] = quantum_score
        
        # Select worker with highest quantum score
        return max(worker_scores.keys(), key=lambda wid: worker_scores[wid])
    
    def assign_task(self, worker_id: str, task_id: str) -> bool:
        """Assign task to worker."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        if worker["current_load"] >= worker["capacity"]:
            return False
        
        worker["current_load"] += 1
        return True
    
    def complete_task(self, worker_id: str, task_id: str, success: bool, response_time_ms: float):
        """Mark task as completed and update worker statistics."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker["current_load"] = max(0, worker["current_load"] - 1)
        
        if success:
            worker["completed_tasks"] += 1
        else:
            worker["failed_tasks"] += 1


# Global instances for production deployment
global_auto_scaler = AutoScalingManager()
global_load_balancer = LoadBalancer()


def async_performance_cache(ttl: int = 3600):
    """Decorator for caching async function results."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = PerformanceCache(default_ttl=ttl)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            cache_key = cache._generate_key(*args, **kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        # Add cache stats to wrapper
        wrapper.cache_stats = lambda: cache.stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


class AutoScalingManager:
    """Advanced auto-scaling manager for quantum-enhanced evaluation systems."""
    
    def __init__(self):
        self.scaling_history = []
        self.performance_thresholds = {
            "cpu_high": 75.0,
            "memory_high": 80.0,
            "response_time_high": 3000,  # ms
            "queue_depth_high": 100,
        }
        self.scaling_factors = {
            "scale_up_cpu": 1.5,
            "scale_up_memory": 1.3,
            "scale_up_latency": 2.0,
            "scale_down": 0.7
        }
        self.min_replicas = 1
        self.max_replicas = 20
        self.cooldown_period = 300  # seconds
        self.last_scaling_action = 0
    
    def analyze_scaling_need(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze current metrics and determine scaling requirements."""
        current_time = time.time()
        
        if current_time - self.last_scaling_action < self.cooldown_period:
            return {"action": "wait", "reason": "Cooldown period active"}
        
        scaling_signals = []
        
        # CPU-based scaling
        if metrics.cpu_usage_percent > self.performance_thresholds["cpu_high"]:
            scaling_signals.append({
                "metric": "cpu",
                "action": "scale_up",
                "factor": self.scaling_factors["scale_up_cpu"]
            })
        
        return self._determine_scaling_action(scaling_signals)
    
    def _determine_scaling_action(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the optimal scaling action from multiple signals."""
        if not signals:
            return {"action": "maintain", "reason": "All metrics within normal range"}
        
        scale_up_signals = [s for s in signals if s["action"] == "scale_up"]
        
        if scale_up_signals:
            most_urgent = scale_up_signals[0]
            return {
                "action": "scale_up",
                "scale_factor": most_urgent["factor"],
                "all_signals": signals
            }
        
        return {"action": "maintain", "reason": "Mixed signals, maintaining current scale"}
    
    def calculate_target_replicas(self, current_replicas: int, scaling_decision: Dict[str, Any]) -> int:
        """Calculate target number of replicas based on scaling decision."""
        if scaling_decision["action"] == "maintain":
            return current_replicas
        
        if scaling_decision["action"] == "scale_up":
            factor = scaling_decision["scale_factor"]
            target = int(current_replicas * factor)
        else:
            return current_replicas
        
        # Apply bounds
        target = max(self.min_replicas, min(target, self.max_replicas))
        
        # Record scaling decision
        self.scaling_history.append({
            "timestamp": time.time(),
            "current_replicas": current_replicas,
            "target_replicas": target,
            "action": scaling_decision["action"]
        })
        
        if scaling_decision["action"] in ["scale_up", "scale_down"]:
            self.last_scaling_action = time.time()
        
        return target


class LoadBalancer:
    """Intelligent load balancer for distributing evaluation tasks."""
    
    def __init__(self):
        self.workers = {}
        self.load_distribution_strategy = "quantum_weighted"
    
    def register_worker(self, worker_id: str, capacity: int, quantum_coherence: float = 1.0):
        """Register a new worker with quantum-enhanced capabilities."""
        self.workers[worker_id] = {
            "capacity": capacity,
            "current_load": 0,
            "quantum_coherence": quantum_coherence,
            "health_status": "healthy",
            "completed_tasks": 0,
            "failed_tasks": 0
        }
    
    def select_optimal_worker(self, task_complexity: float = 0.5) -> Optional[str]:
        """Select optimal worker using quantum-inspired load balancing."""
        healthy_workers = {
            wid: worker for wid, worker in self.workers.items()
            if worker["health_status"] == "healthy"
        }
        
        if not healthy_workers:
            return None
        
        if self.load_distribution_strategy == "quantum_weighted":
            return self._quantum_weighted_selection(healthy_workers, task_complexity)
        else:
            return list(healthy_workers.keys())[0]  # Fallback
    
    def _quantum_weighted_selection(self, workers: Dict[str, Dict], task_complexity: float) -> str:
        """Select worker using quantum-inspired weighting algorithm."""
        worker_scores = {}
        
        for worker_id, worker in workers.items():
            # Base capacity score
            capacity_score = (worker["capacity"] - worker["current_load"]) / worker["capacity"]
            
            # Quantum coherence bonus
            coherence_bonus = worker["quantum_coherence"] * 0.3
            
            # Performance history factor
            if worker["completed_tasks"] > 0:
                success_rate = worker["completed_tasks"] / (worker["completed_tasks"] + worker["failed_tasks"])
                performance_factor = success_rate * 0.4
            else:
                performance_factor = 0.2  # Neutral for new workers
            
            # Quantum superposition of scoring factors
            quantum_score = capacity_score * 0.4 + coherence_bonus + performance_factor
            
            worker_scores[worker_id] = quantum_score
        
        # Select worker with highest quantum score
        return max(worker_scores.keys(), key=lambda wid: worker_scores[wid])
    
    def assign_task(self, worker_id: str, task_id: str) -> bool:
        """Assign task to worker."""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        if worker["current_load"] >= worker["capacity"]:
            return False
        
        worker["current_load"] += 1
        return True
    
    def complete_task(self, worker_id: str, task_id: str, success: bool, response_time_ms: float):
        """Mark task as completed and update worker statistics."""
        if worker_id not in self.workers:
            return
        
        worker = self.workers[worker_id]
        worker["current_load"] = max(0, worker["current_load"] - 1)
        
        if success:
            worker["completed_tasks"] += 1
        else:
            worker["failed_tasks"] += 1


# Global instances for production deployment
global_auto_scaler = AutoScalingManager()
global_load_balancer = LoadBalancer()