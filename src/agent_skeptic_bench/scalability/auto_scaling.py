"""Auto-Scaling and Performance Optimization System.

Provides intelligent auto-scaling, load balancing, resource pooling,
and performance optimization for the Agent Skeptic Bench system.
"""

import asyncio
import logging
import math
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
from asyncio import Queue, Semaphore
from threading import Lock

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predictions
    PROACTIVE = "proactive"  # Scale based on patterns
    QUANTUM_OPTIMIZED = "quantum_optimized"  # Quantum-enhanced scaling


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    QUANTUM_COHERENCE = "quantum_coherence"  # Balance based on quantum metrics


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_rate: float = 0.0
    response_time_p95: float = 0.0
    queue_depth: int = 0
    error_rate: float = 0.0
    quantum_coherence: float = 1.0
    active_connections: int = 0
    throughput: float = 0.0
    
    def get_scaling_score(self) -> float:
        """Calculate overall scaling score (higher = need more resources)."""
        # Weighted combination of metrics
        weights = {
            'cpu_usage': 0.25,
            'memory_usage': 0.20,
            'response_time_p95': 0.20,
            'request_rate': 0.15,
            'queue_depth': 0.10,
            'error_rate': 0.10
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {
            'cpu_usage': min(1.0, self.cpu_usage / 100.0),
            'memory_usage': min(1.0, self.memory_usage / 100.0),
            'response_time_p95': min(1.0, self.response_time_p95 / 5000.0),  # 5s max
            'request_rate': min(1.0, self.request_rate / 1000.0),  # 1000 req/s max
            'queue_depth': min(1.0, self.queue_depth / 1000.0),  # 1000 queue max
            'error_rate': min(1.0, self.error_rate)
        }
        
        score = sum(
            weights[metric] * value
            for metric, value in normalized_metrics.items()
        )
        
        # Quantum coherence adjustment
        if self.quantum_coherence < 0.7:
            score *= 1.2  # Need more resources for low coherence
        
        return min(1.0, score)


@dataclass
class WorkerInstance:
    """Represents a worker instance in the scaling system."""
    worker_id: str
    worker_type: str
    status: str = "starting"  # starting, running, stopping, stopped
    load: float = 0.0
    connections: int = 0
    last_health_check: float = 0.0
    quantum_coherence: float = 1.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    total_requests: int = 0
    
    def get_health_score(self) -> float:
        """Calculate health score for this worker."""
        # Base health score
        health = 1.0
        
        # Reduce for high load
        if self.load > 0.8:
            health *= 0.7
        elif self.load > 0.6:
            health *= 0.9
        
        # Reduce for high error rate
        if self.total_requests > 10:
            error_rate = self.error_count / self.total_requests
            health *= (1.0 - error_rate)
        
        # Reduce for poor response times
        if self.response_times:
            avg_response_time = np.mean(list(self.response_times))
            if avg_response_time > 2000:  # 2 seconds
                health *= 0.8
        
        # Quantum coherence factor
        health *= self.quantum_coherence
        
        return max(0.0, min(1.0, health))


class ResourcePool:
    """Advanced resource pool with dynamic sizing."""
    
    def __init__(self, 
                 pool_name: str,
                 min_size: int = 1,
                 max_size: int = 10,
                 resource_factory: Callable = None):
        """Initialize resource pool."""
        self.pool_name = pool_name
        self.min_size = min_size
        self.max_size = max_size
        self.resource_factory = resource_factory or self._default_resource_factory
        
        self.available_resources = Queue(maxsize=max_size)
        self.all_resources = set()
        self.resource_metrics = defaultdict(lambda: {'usage_count': 0, 'last_used': 0.0})
        self.pool_lock = Lock()
        
        # Initialize minimum resources
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self) -> None:
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            resource = await self.resource_factory()
            self.all_resources.add(resource)
            await self.available_resources.put(resource)
    
    def _default_resource_factory(self) -> Any:
        """Default resource factory."""
        return f"resource_{len(self.all_resources)}"
    
    async def acquire_resource(self, timeout: float = 30.0) -> Any:
        """Acquire a resource from the pool."""
        try:
            # Try to get existing resource
            resource = await asyncio.wait_for(
                self.available_resources.get(),
                timeout=timeout
            )
            
            # Update metrics
            self.resource_metrics[resource]['usage_count'] += 1
            self.resource_metrics[resource]['last_used'] = time.time()
            
            return resource
            
        except asyncio.TimeoutError:
            # Try to create new resource if under max
            if len(self.all_resources) < self.max_size:
                with self.pool_lock:
                    if len(self.all_resources) < self.max_size:
                        resource = await self.resource_factory()
                        self.all_resources.add(resource)
                        self.resource_metrics[resource]['usage_count'] = 1
                        self.resource_metrics[resource]['last_used'] = time.time()
                        return resource
            
            raise RuntimeError(f"Unable to acquire resource from {self.pool_name} pool")
    
    async def release_resource(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        if resource in self.all_resources:
            try:
                await self.available_resources.put(resource)
            except:
                # Pool is full, resource will be garbage collected
                pass
    
    async def cleanup_idle_resources(self, idle_timeout: float = 300.0) -> int:
        """Clean up idle resources beyond minimum size."""
        current_time = time.time()
        cleaned_count = 0
        
        with self.pool_lock:
            if len(self.all_resources) <= self.min_size:
                return cleaned_count
            
            # Find idle resources
            idle_resources = [
                resource for resource in self.all_resources
                if (current_time - self.resource_metrics[resource]['last_used']) > idle_timeout
            ]
            
            # Remove excess idle resources
            excess_count = len(self.all_resources) - self.min_size
            resources_to_remove = idle_resources[:excess_count]
            
            for resource in resources_to_remove:
                self.all_resources.discard(resource)
                self.resource_metrics.pop(resource, None)
                cleaned_count += 1
        
        return cleaned_count
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_name': self.pool_name,
            'total_resources': len(self.all_resources),
            'available_resources': self.available_resources.qsize(),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'utilization_rate': 1.0 - (self.available_resources.qsize() / max(len(self.all_resources), 1))
        }


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME):
        """Initialize load balancer."""
        self.strategy = strategy
        self.workers = {}
        self.round_robin_index = 0
        self.stats_lock = Lock()
    
    def add_worker(self, worker: WorkerInstance) -> None:
        """Add worker to load balancer."""
        with self.stats_lock:
            self.workers[worker.worker_id] = worker
        logger.info(f"Added worker {worker.worker_id} to load balancer")
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove worker from load balancer."""
        with self.stats_lock:
            self.workers.pop(worker_id, None)
        logger.info(f"Removed worker {worker_id} from load balancer")
    
    def select_worker(self) -> Optional[WorkerInstance]:
        """Select best worker based on strategy."""
        with self.stats_lock:
            healthy_workers = [
                worker for worker in self.workers.values()
                if worker.status == "running" and worker.get_health_score() > 0.5
            ]
        
        if not healthy_workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_workers)
        elif self.strategy == LoadBalancingStrategy.QUANTUM_COHERENCE:
            return self._quantum_coherence_selection(healthy_workers)
        
        return healthy_workers[0]  # Fallback
    
    def _round_robin_selection(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Round-robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index = (self.round_robin_index + 1) % len(workers)
        return worker
    
    def _weighted_round_robin_selection(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Weighted round-robin based on health scores."""
        weights = [worker.get_health_score() for worker in workers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return workers[0]
        
        # Weighted random selection
        r = np.random.random() * total_weight
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return workers[i]
        
        return workers[-1]
    
    def _least_connections_selection(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.connections)
    
    def _least_response_time_selection(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker with best response time."""
        def avg_response_time(worker: WorkerInstance) -> float:
            if not worker.response_times:
                return 0.0  # Prefer workers with no history (new workers)
            return np.mean(list(worker.response_times))
        
        return min(workers, key=avg_response_time)
    
    def _quantum_coherence_selection(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker based on quantum coherence and overall health."""
        def quantum_score(worker: WorkerInstance) -> float:
            return worker.quantum_coherence * worker.get_health_score()
        
        return max(workers, key=quantum_score)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.stats_lock:
            total_workers = len(self.workers)
            healthy_workers = sum(
                1 for worker in self.workers.values()
                if worker.status == "running" and worker.get_health_score() > 0.5
            )
            
            total_connections = sum(worker.connections for worker in self.workers.values())
            total_requests = sum(worker.total_requests for worker in self.workers.values())
            total_errors = sum(worker.error_count for worker in self.workers.values())
        
        return {
            'strategy': self.strategy.value,
            'total_workers': total_workers,
            'healthy_workers': healthy_workers,
            'total_connections': total_connections,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_requests, 1),
            'average_load': sum(w.load for w in self.workers.values()) / max(total_workers, 1)
        }


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self, 
                 strategy: ScalingStrategy = ScalingStrategy.QUANTUM_OPTIMIZED,
                 min_workers: int = 2,
                 max_workers: int = 20):
        """Initialize auto-scaler."""
        self.strategy = strategy
        self.min_workers = min_workers
        self.max_workers = max_workers
        
        self.load_balancer = LoadBalancer()
        self.resource_pools = {}
        
        # Scaling decision history
        self.scaling_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=1000)
        
        # Scaling parameters
        self.scale_up_threshold = 0.7
        self.scale_down_threshold = 0.3
        self.scale_cooldown = 60  # seconds
        self.last_scaling_action = 0.0
        
        # Predictive scaling
        self.pattern_detector = self._initialize_pattern_detector()
        
        # Thread pools for different workload types
        self.thread_pools = {
            'cpu_bound': ThreadPoolExecutor(max_workers=4, thread_name_prefix='cpu-'),
            'io_bound': ThreadPoolExecutor(max_workers=10, thread_name_prefix='io-'),
            'evaluation': ProcessPoolExecutor(max_workers=2),  # For CPU-intensive evaluations
        }
        
        # Quantum-enhanced scaling parameters
        self.quantum_scaling_factors = {
            'coherence_threshold': 0.8,
            'entanglement_bonus': 0.1,
            'superposition_factor': 1.2
        }
        
        logger.info(f"Auto-scaler initialized with strategy: {strategy.value}")
    
    def _initialize_pattern_detector(self) -> Dict[str, Any]:
        """Initialize pattern detection for predictive scaling."""
        return {
            'daily_patterns': defaultdict(list),
            'weekly_patterns': defaultdict(list),
            'seasonal_patterns': defaultdict(list),
            'anomaly_threshold': 2.0  # Standard deviations
        }
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # This would integrate with the monitoring system
        # For now, return mock metrics that would be realistic
        import psutil
        
        metrics = ScalingMetrics(
            cpu_usage=psutil.cpu_percent(interval=1),
            memory_usage=psutil.virtual_memory().percent,
            request_rate=len(self.load_balancer.workers) * 10,  # Mock
            response_time_p95=500.0,  # Mock
            queue_depth=5,  # Mock
            error_rate=0.02,  # Mock
            quantum_coherence=0.85,  # Mock
            active_connections=sum(w.connections for w in self.load_balancer.workers.values()),
            throughput=100.0  # Mock
        )
        
        # Store metrics history
        self.metrics_history.append((time.time(), metrics))
        
        return metrics
    
    async def make_scaling_decision(self, metrics: ScalingMetrics) -> Optional[str]:
        """Make scaling decision based on current metrics and strategy."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.scale_cooldown:
            return None
        
        scaling_score = metrics.get_scaling_score()
        current_workers = len(self.load_balancer.workers)
        
        decision = None
        
        if self.strategy == ScalingStrategy.REACTIVE:
            decision = await self._reactive_scaling_decision(scaling_score, current_workers)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            decision = await self._predictive_scaling_decision(metrics, current_workers)
        elif self.strategy == ScalingStrategy.PROACTIVE:
            decision = await self._proactive_scaling_decision(metrics, current_workers)
        elif self.strategy == ScalingStrategy.QUANTUM_OPTIMIZED:
            decision = await self._quantum_optimized_scaling_decision(metrics, current_workers)
        
        if decision:
            self.last_scaling_action = current_time
            self.scaling_history.append({
                'timestamp': current_time,
                'decision': decision,
                'metrics': metrics,
                'workers_before': current_workers,
                'scaling_score': scaling_score
            })
        
        return decision
    
    async def _reactive_scaling_decision(self, scaling_score: float, current_workers: int) -> Optional[str]:
        """Reactive scaling based on current metrics."""
        if scaling_score > self.scale_up_threshold and current_workers < self.max_workers:
            scale_count = min(2, self.max_workers - current_workers)
            return f"scale_up_{scale_count}"
        
        elif scaling_score < self.scale_down_threshold and current_workers > self.min_workers:
            scale_count = min(1, current_workers - self.min_workers)
            return f"scale_down_{scale_count}"
        
        return None
    
    async def _predictive_scaling_decision(self, metrics: ScalingMetrics, current_workers: int) -> Optional[str]:
        """Predictive scaling based on historical patterns."""
        # Analyze patterns and predict future load
        predicted_load = await self._predict_future_load(metrics)
        
        if predicted_load > self.scale_up_threshold * 1.2:  # Scale up proactively
            if current_workers < self.max_workers:
                return "scale_up_1"
        elif predicted_load < self.scale_down_threshold * 0.8:  # Scale down proactively
            if current_workers > self.min_workers:
                return "scale_down_1"
        
        # Fall back to reactive scaling
        return await self._reactive_scaling_decision(metrics.get_scaling_score(), current_workers)
    
    async def _proactive_scaling_decision(self, metrics: ScalingMetrics, current_workers: int) -> Optional[str]:
        """Proactive scaling based on system behavior patterns."""
        # Analyze recent trends
        trend_factor = await self._analyze_metric_trends()
        
        # Adjust thresholds based on trends
        adjusted_up_threshold = self.scale_up_threshold * (1 - trend_factor * 0.2)
        adjusted_down_threshold = self.scale_down_threshold * (1 + trend_factor * 0.2)
        
        scaling_score = metrics.get_scaling_score()
        
        if scaling_score > adjusted_up_threshold and current_workers < self.max_workers:
            return "scale_up_1"
        elif scaling_score < adjusted_down_threshold and current_workers > self.min_workers:
            return "scale_down_1"
        
        return None
    
    async def _quantum_optimized_scaling_decision(self, metrics: ScalingMetrics, current_workers: int) -> Optional[str]:
        """Quantum-enhanced scaling decision with coherence optimization."""
        # Base scaling score
        base_score = metrics.get_scaling_score()
        
        # Quantum enhancement factors
        coherence_factor = metrics.quantum_coherence
        
        # If coherence is low, we might need more resources
        if coherence_factor < self.quantum_scaling_factors['coherence_threshold']:
            # Increase scaling urgency
            quantum_adjusted_score = base_score * self.quantum_scaling_factors['superposition_factor']
        else:
            # System is quantumly coherent, can operate efficiently
            quantum_adjusted_score = base_score * 0.9
        
        # Entanglement bonus - if workers are well-coordinated, we can be more aggressive
        if current_workers > 1:
            worker_health_variance = np.var([
                worker.get_health_score() for worker in self.load_balancer.workers.values()
            ])
            if worker_health_variance < 0.1:  # Low variance = good coordination
                quantum_adjusted_score *= (1 - self.quantum_scaling_factors['entanglement_bonus'])
        
        # Make decision based on quantum-adjusted score
        if quantum_adjusted_score > self.scale_up_threshold and current_workers < self.max_workers:
            # Quantum tunneling effect - sometimes scale up more aggressively
            scale_count = 2 if coherence_factor < 0.7 else 1
            return f"scale_up_{scale_count}"
        
        elif quantum_adjusted_score < self.scale_down_threshold and current_workers > self.min_workers:
            # Quantum interference - be more conservative about scaling down
            if coherence_factor > 0.9:  # Only scale down if very coherent
                return "scale_down_1"
        
        return None
    
    async def _predict_future_load(self, current_metrics: ScalingMetrics) -> float:
        """Predict future load based on historical patterns."""
        if len(self.metrics_history) < 10:
            return current_metrics.get_scaling_score()
        
        # Simple time series prediction using moving average
        recent_scores = [
            metrics.get_scaling_score() for _, metrics in list(self.metrics_history)[-10:]
        ]
        
        # Linear trend calculation
        x = np.arange(len(recent_scores))
        coefficients = np.polyfit(x, recent_scores, 1)
        trend = coefficients[0]
        
        # Predict next value
        predicted_score = recent_scores[-1] + trend
        
        return max(0.0, min(1.0, predicted_score))
    
    async def _analyze_metric_trends(self) -> float:
        """Analyze trends in metrics to inform scaling decisions."""
        if len(self.metrics_history) < 20:
            return 0.0
        
        # Calculate trend factor based on recent metric changes
        recent_metrics = list(self.metrics_history)[-20:]
        
        cpu_trend = self._calculate_trend([m[1].cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m[1].memory_usage for m in recent_metrics])
        response_time_trend = self._calculate_trend([m[1].response_time_p95 for m in recent_metrics])
        
        # Combine trends (positive = increasing load)
        combined_trend = (cpu_trend + memory_trend + response_time_trend) / 3
        
        return np.clip(combined_trend, -1.0, 1.0)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend for a series of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]  # Slope indicates trend
    
    async def execute_scaling_action(self, decision: str) -> bool:
        """Execute the scaling action."""
        try:
            parts = decision.split('_')
            action = parts[0] + '_' + parts[1]  # scale_up or scale_down
            count = int(parts[2]) if len(parts) > 2 else 1
            
            if action == "scale_up":
                return await self._scale_up_workers(count)
            elif action == "scale_down":
                return await self._scale_down_workers(count)
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing scaling action {decision}: {e}")
            return False
    
    async def _scale_up_workers(self, count: int) -> bool:
        """Scale up by adding new workers."""
        for i in range(count):
            worker_id = f"worker_{int(time.time())}_{i}"
            worker = WorkerInstance(
                worker_id=worker_id,
                worker_type="evaluation",
                status="starting"
            )
            
            # Simulate worker startup
            await asyncio.sleep(0.1)  # Mock startup time
            worker.status = "running"
            worker.quantum_coherence = 0.8 + np.random.random() * 0.2  # 0.8-1.0
            
            self.load_balancer.add_worker(worker)
            
            logger.info(f"Scaled up: Added worker {worker_id}")
        
        return True
    
    async def _scale_down_workers(self, count: int) -> bool:
        """Scale down by removing workers."""
        # Select workers to remove (least healthy first)
        workers_to_remove = sorted(
            self.load_balancer.workers.values(),
            key=lambda w: w.get_health_score()
        )[:count]
        
        for worker in workers_to_remove:
            worker.status = "stopping"
            
            # Wait for current connections to finish (simplified)
            await asyncio.sleep(0.1)
            
            self.load_balancer.remove_worker(worker.worker_id)
            
            logger.info(f"Scaled down: Removed worker {worker.worker_id}")
        
        return True
    
    async def optimize_resource_pools(self) -> Dict[str, Any]:
        """Optimize resource pools based on usage patterns."""
        optimization_results = {}
        
        for pool_name, pool in self.resource_pools.items():
            stats = pool.get_pool_stats()
            
            # Clean up idle resources
            cleaned_count = await pool.cleanup_idle_resources()
            
            # Adjust pool size based on utilization
            if stats['utilization_rate'] > 0.9 and stats['total_resources'] < pool.max_size:
                # Pool is highly utilized, consider increasing max size
                new_max_size = min(pool.max_size + 2, 20)
                pool.max_size = new_max_size
                optimization_results[pool_name] = f"Increased max size to {new_max_size}"
            
            elif stats['utilization_rate'] < 0.3 and stats['total_resources'] > pool.min_size:
                # Pool is underutilized, consider decreasing min size
                new_min_size = max(pool.min_size - 1, 1)
                pool.min_size = new_min_size
                optimization_results[pool_name] = f"Decreased min size to {new_min_size}"
            
            else:
                optimization_results[pool_name] = f"Cleaned {cleaned_count} idle resources"
        
        return optimization_results
    
    async def get_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling report."""
        current_metrics = await self.collect_metrics()
        
        # Calculate scaling efficiency
        efficiency_metrics = await self._calculate_scaling_efficiency()
        
        # Analyze scaling history
        scaling_analysis = await self._analyze_scaling_history()
        
        return {
            'timestamp': time.time(),
            'strategy': self.strategy.value,
            'current_metrics': current_metrics,
            'current_workers': len(self.load_balancer.workers),
            'worker_range': {'min': self.min_workers, 'max': self.max_workers},
            'load_balancer_stats': self.load_balancer.get_load_balancer_stats(),
            'efficiency_metrics': efficiency_metrics,
            'scaling_analysis': scaling_analysis,
            'resource_pools': {
                name: pool.get_pool_stats() 
                for name, pool in self.resource_pools.items()
            },
            'recommendations': await self._generate_scaling_recommendations()
        }
    
    async def _calculate_scaling_efficiency(self) -> Dict[str, float]:
        """Calculate scaling efficiency metrics."""
        if not self.scaling_history:
            return {'efficiency_score': 1.0}
        
        recent_actions = list(self.scaling_history)[-10:]  # Last 10 actions
        
        # Calculate success rate (mock implementation)
        successful_actions = sum(
            1 for action in recent_actions
            if 'successful' not in action or action.get('successful', True)
        )
        
        success_rate = successful_actions / len(recent_actions)
        
        # Calculate response time improvement
        response_time_improvement = 0.0
        if len(recent_actions) > 1:
            before_metrics = [action['metrics'] for action in recent_actions]
            avg_before = np.mean([m.response_time_p95 for m in before_metrics[:-1]])
            avg_after = np.mean([m.response_time_p95 for m in before_metrics[1:]])
            response_time_improvement = max(0, (avg_before - avg_after) / avg_before)
        
        # Calculate resource utilization efficiency
        utilization_efficiency = 0.8  # Mock value
        
        efficiency_score = (success_rate + response_time_improvement + utilization_efficiency) / 3
        
        return {
            'efficiency_score': efficiency_score,
            'success_rate': success_rate,
            'response_time_improvement': response_time_improvement,
            'utilization_efficiency': utilization_efficiency
        }
    
    async def _analyze_scaling_history(self) -> Dict[str, Any]:
        """Analyze scaling decision history."""
        if not self.scaling_history:
            return {}
        
        scale_up_count = sum(1 for action in self.scaling_history if 'scale_up' in action['decision'])
        scale_down_count = sum(1 for action in self.scaling_history if 'scale_down' in action['decision'])
        
        avg_scaling_score = np.mean([action['scaling_score'] for action in self.scaling_history])
        
        return {
            'total_scaling_actions': len(self.scaling_history),
            'scale_up_actions': scale_up_count,
            'scale_down_actions': scale_down_count,
            'average_scaling_score': avg_scaling_score,
            'scaling_frequency_per_hour': len(self.scaling_history) / max(1, 
                (time.time() - self.scaling_history[0]['timestamp']) / 3600
            )
        }
    
    async def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling optimization recommendations."""
        recommendations = []
        
        current_workers = len(self.load_balancer.workers)
        
        # Check if scaling bounds are appropriate
        if current_workers == self.max_workers:
            recommendations.append(
                "System is at maximum worker capacity. Consider increasing max_workers limit."
            )
        
        if current_workers == self.min_workers and len(self.scaling_history) > 5:
            recent_scale_ups = sum(
                1 for action in list(self.scaling_history)[-5:]
                if 'scale_up' in action['decision']
            )
            if recent_scale_ups > 3:
                recommendations.append(
                    "Frequent scale-up events detected. Consider increasing min_workers."
                )
        
        # Analyze efficiency
        efficiency_metrics = await self._calculate_scaling_efficiency()
        if efficiency_metrics['efficiency_score'] < 0.7:
            recommendations.append(
                "Scaling efficiency is below optimal. Review scaling thresholds and strategy."
            )
        
        # Check quantum coherence impact
        if self.strategy == ScalingStrategy.QUANTUM_OPTIMIZED:
            avg_coherence = np.mean([
                w.quantum_coherence for w in self.load_balancer.workers.values()
            ]) if self.load_balancer.workers else 1.0
            
            if avg_coherence < 0.8:
                recommendations.append(
                    "Low quantum coherence detected. Consider quantum optimization or worker rebalancing."
                )
        
        if not recommendations:
            recommendations.append("Scaling system is operating efficiently. Continue monitoring.")
        
        return recommendations
    
    async def start_auto_scaling_loop(self) -> None:
        """Start the auto-scaling monitoring loop."""
        logger.info("Starting auto-scaling loop")
        
        while True:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Make scaling decision
                decision = await self.make_scaling_decision(metrics)
                
                if decision:
                    logger.info(f"Scaling decision: {decision}")
                    success = await self.execute_scaling_action(decision)
                    if success:
                        logger.info(f"Successfully executed scaling action: {decision}")
                    else:
                        logger.error(f"Failed to execute scaling action: {decision}")
                
                # Optimize resource pools
                await self.optimize_resource_pools()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30-second monitoring interval
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(30)
    
    def create_resource_pool(self, 
                           pool_name: str,
                           min_size: int = 1,
                           max_size: int = 10,
                           resource_factory: Callable = None) -> ResourcePool:
        """Create and register a new resource pool."""
        pool = ResourcePool(pool_name, min_size, max_size, resource_factory)
        self.resource_pools[pool_name] = pool
        return pool
    
    async def shutdown(self) -> None:
        """Graceful shutdown of auto-scaling system."""
        logger.info("Shutting down auto-scaling system")
        
        # Shutdown thread pools
        for pool_name, pool in self.thread_pools.items():
            pool.shutdown(wait=True)
            logger.info(f"Shutdown thread pool: {pool_name}")
        
        # Clean up workers
        for worker in list(self.load_balancer.workers.values()):
            worker.status = "stopping"
            self.load_balancer.remove_worker(worker.worker_id)
        
        logger.info("Auto-scaling system shutdown complete")
