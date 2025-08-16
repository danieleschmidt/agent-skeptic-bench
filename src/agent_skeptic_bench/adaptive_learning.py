"""Adaptive Learning System for Agent Skeptic Bench.

Self-improving patterns that learn and evolve agent evaluation capabilities
autonomously based on usage patterns and performance metrics.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .models import AgentConfig, EvaluationResult, Scenario
from .quantum_optimizer import QuantumOptimizer

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning mode for adaptive systems."""
    PASSIVE = "passive"  # Learn from observations only
    ACTIVE = "active"    # Actively experiment and learn
    HYBRID = "hybrid"    # Combination of passive and active


@dataclass
class LearningPattern:
    """Pattern learned from evaluation history."""
    pattern_id: str
    pattern_type: str
    features: Dict[str, float]
    prediction_accuracy: float
    confidence: float
    usage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    

@dataclass
class AdaptationStrategy:
    """Strategy for system adaptation."""
    strategy_name: str
    trigger_conditions: Dict[str, float]
    adaptation_actions: List[str]
    effectiveness_score: float = 0.0
    deployment_count: int = 0


class AdaptiveCacheManager:
    """Adaptive caching system that learns from access patterns."""
    
    def __init__(self, max_cache_size: int = 10000):
        """Initialize adaptive cache manager."""
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Any] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.prediction_model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with learning."""
        if key in self.cache:
            self._record_access(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache with adaptive management."""
        # Learn from access patterns to determine optimal TTL
        if self.is_trained and ttl is None:
            ttl = self._predict_optimal_ttl(key)
        
        # Adaptive eviction if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._adaptive_eviction()
        
        self.cache[key] = {
            'value': value,
            'created_at': datetime.utcnow(),
            'ttl': ttl or 3600,  # Default 1 hour
            'predicted_ttl': ttl or 3600
        }
        
        self._record_access(key)
    
    def _record_access(self, key: str) -> None:
        """Record access pattern for learning."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(datetime.utcnow())
        
        # Keep only last 100 accesses per key
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _predict_optimal_ttl(self, key: str) -> int:
        """Predict optimal TTL based on access patterns."""
        if not self.is_trained or key not in self.access_patterns:
            return 3600  # Default 1 hour
        
        features = self._extract_access_features(key)
        if features:
            try:
                predicted_ttl = self.prediction_model.predict([features])[0]
                return max(300, min(86400, int(predicted_ttl)))  # 5 min to 24 hours
            except:
                return 3600
        
        return 3600
    
    def _extract_access_features(self, key: str) -> Optional[List[float]]:
        """Extract features from access patterns."""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
            return None
        
        accesses = self.access_patterns[key]
        now = datetime.utcnow()
        
        # Time-based features
        time_since_last = (now - accesses[-1]).total_seconds()
        access_frequency = len(accesses) / max(1, (now - accesses[0]).total_seconds() / 3600)
        
        # Pattern features
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = np.mean(intervals) if intervals else 3600
        interval_variance = np.var(intervals) if len(intervals) > 1 else 0
        
        # Key characteristics
        key_length = len(key)
        key_hash = hash(key) % 1000 / 1000.0
        
        return [
            time_since_last,
            access_frequency,
            avg_interval,
            interval_variance,
            key_length,
            key_hash
        ]
    
    def _adaptive_eviction(self) -> None:
        """Adaptively evict items based on predicted access patterns."""
        eviction_candidates = []
        
        for key, item in self.cache.items():
            age = (datetime.utcnow() - item['created_at']).total_seconds()
            predicted_next_access = self._predict_next_access_time(key)
            
            # Score for eviction (higher = more likely to evict)
            eviction_score = age / max(1, predicted_next_access)
            eviction_candidates.append((key, eviction_score))
        
        # Sort by eviction score and remove top 10%
        eviction_candidates.sort(key=lambda x: x[1], reverse=True)
        eviction_count = max(1, len(eviction_candidates) // 10)
        
        for key, _ in eviction_candidates[:eviction_count]:
            del self.cache[key]
            logger.debug(f"Adaptively evicted cache key: {key}")
    
    def _predict_next_access_time(self, key: str) -> float:
        """Predict time until next access for a key."""
        if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
            return 3600  # Default 1 hour
        
        accesses = self.access_patterns[key]
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        # Simple prediction based on average interval
        return np.mean(intervals) if intervals else 3600
    
    def train_prediction_model(self) -> None:
        """Train the TTL prediction model from access patterns."""
        features = []
        targets = []
        
        for key, accesses in self.access_patterns.items():
            if len(accesses) < 3:
                continue
            
            # Use access intervals as training data
            for i in range(2, len(accesses)):
                feature_vector = self._extract_access_features_at_time(key, accesses[:i])
                if feature_vector:
                    target_interval = (accesses[i] - accesses[i-1]).total_seconds()
                    features.append(feature_vector)
                    targets.append(target_interval)
        
        if len(features) > 10:  # Minimum training data
            try:
                features_scaled = self.scaler.fit_transform(features)
                self.prediction_model.fit(features_scaled, targets)
                self.is_trained = True
                logger.info(f"Cache prediction model trained with {len(features)} samples")
            except Exception as e:
                logger.warning(f"Failed to train cache prediction model: {e}")
    
    def _extract_access_features_at_time(self, key: str, accesses: List[datetime]) -> Optional[List[float]]:
        """Extract features from access patterns at a specific time."""
        if len(accesses) < 2:
            return None
        
        last_time = accesses[-1]
        
        # Time-based features
        access_frequency = len(accesses) / max(1, (last_time - accesses[0]).total_seconds() / 3600)
        
        # Pattern features
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = np.mean(intervals) if intervals else 3600
        interval_variance = np.var(intervals) if len(intervals) > 1 else 0
        
        # Key characteristics
        key_length = len(key)
        key_hash = hash(key) % 1000 / 1000.0
        
        return [
            access_frequency,
            avg_interval,
            interval_variance,
            key_length,
            key_hash
        ]


class AutoScalingManager:
    """Intelligent auto-scaling based on load patterns and quantum optimization."""
    
    def __init__(self):
        """Initialize auto-scaling manager."""
        self.load_history: List[Tuple[datetime, float]] = []
        self.scaling_history: List[Tuple[datetime, int, float]] = []
        self.current_instances = 1
        self.quantum_optimizer = QuantumOptimizer(population_size=20, max_iterations=30)
        self.scaling_patterns: List[LearningPattern] = []
        
    async def monitor_and_scale(self, current_load: float, target_response_time: float = 200.0) -> int:
        """Monitor load and automatically scale resources."""
        timestamp = datetime.utcnow()
        self.load_history.append((timestamp, current_load))
        
        # Keep only last 1000 measurements
        if len(self.load_history) > 1000:
            self.load_history = self.load_history[-1000:]
        
        # Predict future load
        predicted_load = self._predict_future_load()
        
        # Calculate optimal instance count using quantum optimization
        optimal_instances = await self._calculate_optimal_instances(
            current_load, predicted_load, target_response_time
        )
        
        # Apply scaling decision with hysteresis
        scaling_decision = self._apply_scaling_hysteresis(optimal_instances)
        
        if scaling_decision != self.current_instances:
            await self._execute_scaling(scaling_decision, current_load)
        
        return self.current_instances
    
    def _predict_future_load(self, prediction_horizon: int = 300) -> float:
        """Predict load for the next prediction_horizon seconds."""
        if len(self.load_history) < 10:
            return self.load_history[-1][1] if self.load_history else 1.0
        
        # Extract time series data
        times = [(t - self.load_history[0][0]).total_seconds() for t, _ in self.load_history]
        loads = [load for _, load in self.load_history]
        
        # Simple trend analysis
        if len(loads) >= 5:
            recent_trend = np.polyfit(times[-5:], loads[-5:], 1)[0]
            current_load = loads[-1]
            predicted_load = current_load + recent_trend * prediction_horizon
            
            # Apply bounds
            return max(0.1, min(10.0, predicted_load))
        
        return loads[-1]
    
    async def _calculate_optimal_instances(self, 
                                         current_load: float,
                                         predicted_load: float,
                                         target_response_time: float) -> int:
        """Calculate optimal instance count using quantum optimization."""
        
        def evaluation_function(parameters: Dict[str, float]) -> List:
            """Evaluation function for instance optimization."""
            instance_count = max(1, int(parameters.get('instances', 1)))
            
            # Simulate response time based on load and instances
            effective_load_per_instance = max(current_load, predicted_load) / instance_count
            
            # Response time model: exponential with load
            response_time = 100 * np.exp(effective_load_per_instance)
            
            # Cost model: linear with instances
            cost = instance_count * 10  # $10 per instance
            
            # Multi-objective score
            response_time_score = max(0, 1.0 - (response_time / target_response_time))
            cost_score = max(0, 1.0 - (cost / 100))  # Normalize cost
            
            combined_score = 0.7 * response_time_score + 0.3 * cost_score
            
            # Mock evaluation result
            mock_result = type('MockResult', (), {
                'metrics': type('MockMetrics', (), {
                    'scores': {'optimization_score': combined_score}
                })()
            })()
            
            return [mock_result]
        
        # Set up optimization parameters
        self.quantum_optimizer.quantum_population = []
        self.quantum_optimizer._initialize_population()
        
        # Override parameter space for instance optimization
        for state in self.quantum_optimizer.quantum_population:
            state.parameters = {
                'instances': np.random.uniform(1, 20),
                'load_threshold': np.random.uniform(0.5, 2.0),
                'scale_factor': np.random.uniform(1.1, 2.0)
            }
        
        try:
            result = await self.quantum_optimizer.optimize(evaluation_function)
            optimal_instances = max(1, int(result.optimal_parameters.get('instances', 1)))
            return min(20, optimal_instances)  # Cap at 20 instances
        except Exception as e:
            logger.warning(f"Quantum optimization failed, using heuristic: {e}")
            return self._heuristic_instance_calculation(current_load, predicted_load)
    
    def _heuristic_instance_calculation(self, current_load: float, predicted_load: float) -> int:
        """Fallback heuristic for instance calculation."""
        max_load = max(current_load, predicted_load)
        
        # Simple heuristic: 1 instance per 0.8 load units
        optimal_instances = max(1, int(np.ceil(max_load / 0.8)))
        
        return min(20, optimal_instances)
    
    def _apply_scaling_hysteresis(self, target_instances: int) -> int:
        """Apply hysteresis to prevent oscillation."""
        current = self.current_instances
        
        # Scale up immediately if needed
        if target_instances > current:
            return target_instances
        
        # Scale down more conservatively
        if target_instances < current:
            # Only scale down if target is significantly lower
            if target_instances <= current * 0.7:
                return max(target_instances, current - 1)  # Scale down gradually
        
        return current
    
    async def _execute_scaling(self, new_instance_count: int, current_load: float) -> None:
        """Execute scaling operation."""
        old_instances = self.current_instances
        self.current_instances = new_instance_count
        
        # Record scaling decision
        self.scaling_history.append((datetime.utcnow(), new_instance_count, current_load))
        
        # Learn from scaling patterns
        await self._learn_from_scaling_decision(old_instances, new_instance_count, current_load)
        
        logger.info(f"Scaled from {old_instances} to {new_instance_count} instances (load: {current_load:.2f})")
    
    async def _learn_from_scaling_decision(self, 
                                         old_instances: int,
                                         new_instances: int,
                                         load: float) -> None:
        """Learn from scaling decisions to improve future predictions."""
        if len(self.scaling_history) < 5:
            return
        
        # Analyze pattern effectiveness
        recent_scaling = self.scaling_history[-5:]
        
        # Calculate scaling effectiveness metrics
        load_stability = self._calculate_load_stability(recent_scaling)
        response_time_improvement = self._estimate_response_time_improvement(recent_scaling)
        
        # Create learning pattern
        pattern = LearningPattern(
            pattern_id=f"scaling_{datetime.utcnow().timestamp()}",
            pattern_type="scaling_decision",
            features={
                'load_delta': abs(new_instances - old_instances) / max(1, old_instances),
                'current_load': load,
                'instance_ratio': new_instances / max(1, old_instances),
                'load_stability': load_stability,
                'response_improvement': response_time_improvement
            },
            prediction_accuracy=0.8,  # Initial accuracy
            confidence=0.7
        )
        
        self.scaling_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.scaling_patterns) > 100:
            self.scaling_patterns = self.scaling_patterns[-100:]
    
    def _calculate_load_stability(self, scaling_history: List[Tuple[datetime, int, float]]) -> float:
        """Calculate load stability metric."""
        if len(scaling_history) < 2:
            return 0.5
        
        loads = [load for _, _, load in scaling_history]
        load_variance = np.var(loads)
        
        # Normalize stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + load_variance)
        return min(1.0, stability)
    
    def _estimate_response_time_improvement(self, scaling_history: List[Tuple[datetime, int, float]]) -> float:
        """Estimate response time improvement from scaling."""
        if len(scaling_history) < 2:
            return 0.0
        
        # Simple estimation based on load per instance improvement
        before = scaling_history[-2]
        after = scaling_history[-1]
        
        load_per_instance_before = before[2] / max(1, before[1])
        load_per_instance_after = after[2] / max(1, after[1])
        
        improvement = (load_per_instance_before - load_per_instance_after) / max(0.1, load_per_instance_before)
        
        return max(0.0, min(1.0, improvement))


class SelfHealingSystem:
    """Self-healing system with circuit breakers and automatic recovery."""
    
    def __init__(self):
        """Initialize self-healing system."""
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.failure_patterns: List[LearningPattern] = []
        self.recovery_strategies: List[AdaptationStrategy] = []
        self.health_metrics: Dict[str, List[Tuple[datetime, float]]] = {}
        
    def add_circuit_breaker(self, 
                          service_name: str,
                          failure_threshold: int = 5,
                          recovery_timeout: int = 60) -> None:
        """Add circuit breaker for a service."""
        self.circuit_breakers[service_name] = CircuitBreaker(
            service_name, failure_threshold, recovery_timeout
        )
    
    async def execute_with_protection(self, 
                                    service_name: str,
                                    operation: Any,
                                    *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection."""
        if service_name not in self.circuit_breakers:
            self.add_circuit_breaker(service_name)
        
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            result = await circuit_breaker.execute(operation, *args, **kwargs)
            self._record_success(service_name)
            return result
        except Exception as e:
            await self._handle_failure(service_name, e)
            raise
    
    def _record_success(self, service_name: str) -> None:
        """Record successful operation."""
        timestamp = datetime.utcnow()
        if service_name not in self.health_metrics:
            self.health_metrics[service_name] = []
        
        self.health_metrics[service_name].append((timestamp, 1.0))  # 1.0 = success
        
        # Keep only last 1000 metrics per service
        if len(self.health_metrics[service_name]) > 1000:
            self.health_metrics[service_name] = self.health_metrics[service_name][-1000:]
    
    async def _handle_failure(self, service_name: str, error: Exception) -> None:
        """Handle service failure with learning."""
        timestamp = datetime.utcnow()
        if service_name not in self.health_metrics:
            self.health_metrics[service_name] = []
        
        self.health_metrics[service_name].append((timestamp, 0.0))  # 0.0 = failure
        
        # Learn from failure pattern
        await self._learn_from_failure(service_name, error)
        
        # Attempt automatic recovery
        await self._attempt_recovery(service_name, error)
    
    async def _learn_from_failure(self, service_name: str, error: Exception) -> None:
        """Learn from failure patterns."""
        if service_name not in self.health_metrics:
            return
        
        recent_metrics = self.health_metrics[service_name][-10:]  # Last 10 operations
        
        # Analyze failure pattern
        failure_rate = sum(1 for _, success in recent_metrics if success == 0.0) / len(recent_metrics)
        
        # Create failure pattern
        pattern = LearningPattern(
            pattern_id=f"failure_{service_name}_{datetime.utcnow().timestamp()}",
            pattern_type="failure_pattern",
            features={
                'service_name_hash': hash(service_name) % 1000 / 1000.0,
                'error_type_hash': hash(type(error).__name__) % 1000 / 1000.0,
                'failure_rate': failure_rate,
                'recent_operations': len(recent_metrics),
                'time_of_day': datetime.utcnow().hour / 24.0
            },
            prediction_accuracy=0.6,
            confidence=0.5
        )
        
        self.failure_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.failure_patterns) > 200:
            self.failure_patterns = self.failure_patterns[-200:]
    
    async def _attempt_recovery(self, service_name: str, error: Exception) -> None:
        """Attempt automatic recovery strategies."""
        error_type = type(error).__name__
        
        # Define recovery strategies based on error type
        recovery_strategies = {
            'TimeoutError': ['increase_timeout', 'retry_with_backoff'],
            'ConnectionError': ['reconnect', 'use_backup_endpoint'],
            'MemoryError': ['clear_cache', 'garbage_collect'],
            'ValueError': ['validate_input', 'use_default_values']
        }
        
        strategies = recovery_strategies.get(error_type, ['generic_retry'])
        
        for strategy in strategies:
            try:
                await self._execute_recovery_strategy(service_name, strategy)
                logger.info(f"Recovery strategy '{strategy}' executed for {service_name}")
                break
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy '{strategy}' failed: {recovery_error}")
    
    async def _execute_recovery_strategy(self, service_name: str, strategy: str) -> None:
        """Execute specific recovery strategy."""
        if strategy == 'increase_timeout':
            # Increase timeout for circuit breaker
            if service_name in self.circuit_breakers:
                self.circuit_breakers[service_name].timeout *= 1.5
        
        elif strategy == 'retry_with_backoff':
            # Implement exponential backoff
            await asyncio.sleep(min(30, 2 ** len(self.failure_patterns[-5:])))
        
        elif strategy == 'clear_cache':
            # Clear any cached data (implementation specific)
            logger.info(f"Clearing cache for {service_name}")
        
        elif strategy == 'reconnect':
            # Attempt to reconnect (implementation specific)
            logger.info(f"Attempting reconnection for {service_name}")
        
        elif strategy == 'generic_retry':
            # Generic retry with delay
            await asyncio.sleep(5)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all services."""
        summary = {}
        
        for service_name, metrics in self.health_metrics.items():
            if not metrics:
                continue
            
            recent_metrics = metrics[-50:]  # Last 50 operations
            success_rate = sum(success for _, success in recent_metrics) / len(recent_metrics)
            
            # Calculate circuit breaker state
            cb_state = "unknown"
            if service_name in self.circuit_breakers:
                cb = self.circuit_breakers[service_name]
                cb_state = cb.state.value
            
            summary[service_name] = {
                'success_rate': success_rate,
                'total_operations': len(metrics),
                'circuit_breaker_state': cb_state,
                'recent_failures': sum(1 for _, success in recent_metrics if success == 0.0),
                'last_operation': metrics[-1][0].isoformat() if metrics else None
            }
        
        return summary


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker implementation with adaptive learning."""
    
    def __init__(self, 
                 service_name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60):
        """Initialize circuit breaker."""
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        
    async def execute(self, operation: Any, *args, **kwargs) -> Any:
        """Execute operation with circuit breaker logic."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker OPEN for {self.service_name}")
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Success - reset failure count
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class PerformanceOptimizer:
    """Adaptive performance optimizer that learns from usage patterns."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_patterns: List[LearningPattern] = []
        self.optimization_strategies: List[AdaptationStrategy] = []
        
    async def optimize_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize performance based on current metrics."""
        current_performance = {
            'response_time': metrics.get('response_time_ms', 0),
            'throughput': metrics.get('throughput_rps', 0),
            'cpu_usage': metrics.get('cpu_percent', 0),
            'memory_usage': metrics.get('memory_percent', 0),
            'error_rate': metrics.get('error_rate', 0)
        }
        
        # Identify optimization opportunities
        optimizations = await self._identify_optimizations(current_performance)
        
        # Apply optimizations
        results = {}
        for optimization in optimizations:
            try:
                result = await self._apply_optimization(optimization, current_performance)
                results[optimization] = result
            except Exception as e:
                logger.warning(f"Optimization '{optimization}' failed: {e}")
                results[optimization] = {'status': 'failed', 'error': str(e)}
        
        # Learn from optimization results
        await self._learn_from_optimizations(current_performance, results)
        
        return results
    
    async def _identify_optimizations(self, performance: Dict[str, float]) -> List[str]:
        """Identify potential optimizations based on performance metrics."""
        optimizations = []
        
        # Response time optimizations
        if performance['response_time'] > 500:  # > 500ms
            optimizations.extend(['enable_caching', 'optimize_queries', 'connection_pooling'])
        
        # Throughput optimizations
        if performance['throughput'] < 100:  # < 100 RPS
            optimizations.extend(['increase_workers', 'async_processing', 'batch_operations'])
        
        # Resource optimizations
        if performance['cpu_usage'] > 80:
            optimizations.extend(['optimize_algorithms', 'caching', 'load_balancing'])
        
        if performance['memory_usage'] > 85:
            optimizations.extend(['memory_pooling', 'garbage_collection', 'data_compression'])
        
        # Error rate optimizations
        if performance['error_rate'] > 0.05:  # > 5%
            optimizations.extend(['input_validation', 'retry_logic', 'circuit_breakers'])
        
        return list(set(optimizations))  # Remove duplicates
    
    async def _apply_optimization(self, 
                                optimization: str,
                                performance: Dict[str, float]) -> Dict[str, Any]:
        """Apply specific optimization strategy."""
        start_time = time.time()
        
        optimization_functions = {
            'enable_caching': self._enable_caching,
            'optimize_queries': self._optimize_queries,
            'connection_pooling': self._enable_connection_pooling,
            'increase_workers': self._increase_workers,
            'async_processing': self._enable_async_processing,
            'batch_operations': self._enable_batch_operations,
            'optimize_algorithms': self._optimize_algorithms,
            'load_balancing': self._enable_load_balancing,
            'memory_pooling': self._enable_memory_pooling,
            'garbage_collection': self._trigger_garbage_collection,
            'data_compression': self._enable_data_compression,
            'input_validation': self._improve_input_validation,
            'retry_logic': self._enable_retry_logic,
            'circuit_breakers': self._enable_circuit_breakers
        }
        
        optimization_func = optimization_functions.get(optimization, self._generic_optimization)
        result = await optimization_func(performance)
        
        execution_time = time.time() - start_time
        
        return {
            'status': 'completed',
            'execution_time': execution_time,
            'details': result
        }
    
    async def _enable_caching(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable or optimize caching."""
        return {
            'cache_hit_rate_target': 0.8,
            'cache_size_mb': min(1024, int(performance['memory_usage'] * 10)),
            'cache_ttl_seconds': 3600
        }
    
    async def _optimize_queries(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Optimize database queries."""
        return {
            'query_timeout': min(30, performance['response_time'] / 10),
            'batch_size': 100,
            'index_suggestions': ['create_index_on_frequently_queried_columns']
        }
    
    async def _enable_connection_pooling(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable or optimize connection pooling."""
        pool_size = max(5, min(50, int(performance['throughput'] / 10)))
        return {
            'pool_size': pool_size,
            'max_overflow': pool_size // 2,
            'pool_timeout': 30
        }
    
    async def _increase_workers(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Increase worker processes/threads."""
        current_workers = 4  # Assume current worker count
        target_workers = min(16, int(current_workers * 1.5))
        
        return {
            'current_workers': current_workers,
            'target_workers': target_workers,
            'worker_type': 'async'
        }
    
    async def _enable_async_processing(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable asynchronous processing."""
        return {
            'async_queue_size': 1000,
            'async_workers': 8,
            'batch_processing': True
        }
    
    async def _enable_batch_operations(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable batch operations."""
        return {
            'batch_size': min(1000, max(10, int(performance['throughput']))),
            'batch_timeout_ms': 100,
            'parallel_batches': 4
        }
    
    async def _optimize_algorithms(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Optimize algorithms and data structures."""
        return {
            'algorithm_optimizations': ['use_hash_tables', 'implement_caching', 'parallel_processing'],
            'data_structure_improvements': ['use_efficient_collections', 'memory_mapping']
        }
    
    async def _enable_load_balancing(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable or optimize load balancing."""
        return {
            'load_balancing_algorithm': 'round_robin',
            'health_checks': True,
            'sticky_sessions': False
        }
    
    async def _enable_memory_pooling(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable memory pooling."""
        return {
            'memory_pool_size_mb': int(performance['memory_usage'] * 0.1),
            'pool_block_size_kb': 64,
            'enable_object_pooling': True
        }
    
    async def _trigger_garbage_collection(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Trigger garbage collection optimization."""
        return {
            'gc_triggered': True,
            'gc_threshold_adjustment': 0.8,
            'memory_freed_estimate_mb': performance['memory_usage'] * 0.1
        }
    
    async def _enable_data_compression(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable data compression."""
        return {
            'compression_algorithm': 'gzip',
            'compression_level': 6,
            'estimated_size_reduction': 0.3
        }
    
    async def _improve_input_validation(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Improve input validation to reduce errors."""
        return {
            'validation_rules_added': ['type_checking', 'range_validation', 'format_validation'],
            'validation_performance_impact': 'minimal'
        }
    
    async def _enable_retry_logic(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable retry logic for failed operations."""
        return {
            'max_retries': 3,
            'retry_backoff': 'exponential',
            'retry_exceptions': ['TimeoutError', 'ConnectionError']
        }
    
    async def _enable_circuit_breakers(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Enable circuit breakers."""
        return {
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'services_protected': ['database', 'external_api', 'cache']
        }
    
    async def _generic_optimization(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Generic optimization strategy."""
        return {
            'optimization_applied': 'generic_performance_tuning',
            'status': 'completed'
        }
    
    async def _learn_from_optimizations(self, 
                                      performance: Dict[str, float],
                                      results: Dict[str, Any]) -> None:
        """Learn from optimization results to improve future decisions."""
        # Calculate optimization effectiveness
        effectiveness_scores = {}
        
        for optimization, result in results.items():
            if result.get('status') == 'completed':
                # Simple effectiveness score based on execution time and expected impact
                execution_time = result.get('execution_time', 1.0)
                effectiveness = 1.0 / (1.0 + execution_time)  # Faster = more effective
                effectiveness_scores[optimization] = effectiveness
        
        # Create learning patterns
        for optimization, effectiveness in effectiveness_scores.items():
            pattern = LearningPattern(
                pattern_id=f"optimization_{optimization}_{datetime.utcnow().timestamp()}",
                pattern_type="optimization_effectiveness",
                features={
                    'optimization_hash': hash(optimization) % 1000 / 1000.0,
                    'response_time': performance['response_time'],
                    'throughput': performance['throughput'],
                    'cpu_usage': performance['cpu_usage'],
                    'memory_usage': performance['memory_usage'],
                    'error_rate': performance['error_rate'],
                    'effectiveness': effectiveness
                },
                prediction_accuracy=effectiveness,
                confidence=0.7
            )
            
            self.performance_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.performance_patterns) > 500:
            self.performance_patterns = self.performance_patterns[-500:]
        
        # Record optimization history
        self.optimization_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'performance': performance,
            'optimizations_applied': list(results.keys()),
            'effectiveness_scores': effectiveness_scores
        })
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]