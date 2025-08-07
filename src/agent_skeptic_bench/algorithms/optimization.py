"""Performance optimization algorithms for Agent Skeptic Bench with quantum-inspired enhancements."""

import logging
import asyncio
import time
import math
import random
import numpy as np
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from datetime import datetime, timedelta

from ..models import (
    EvaluationResult, 
    Scenario, 
    EvaluationMetrics, 
    SkepticResponse, 
    QuantumOptimizationState,
    AdaptationMetrics
)


logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Available cache backends."""
    
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    
    LAZY_LOADING = "lazy_loading"
    PREFETCHING = "prefetching"
    BATCH_PROCESSING = "batch_processing"
    PARALLEL_EXECUTION = "parallel_execution"
    CACHING = "caching"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    
    execution_time: float
    memory_usage: float
    cache_hit_rate: float
    database_queries: int
    api_calls: int
    concurrent_operations: int
    throughput: float
    latency_p95: float
    error_rate: float


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    enable_caching: bool = True
    cache_backend: CacheBackend = CacheBackend.MEMORY
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000
    enable_prefetching: bool = True
    prefetch_batch_size: int = 10
    enable_parallel_execution: bool = True
    max_workers: int = 4
    batch_size: int = 50
    optimization_strategies: List[OptimizationStrategy] = field(default_factory=lambda: [
        OptimizationStrategy.CACHING,
        OptimizationStrategy.PARALLEL_EXECUTION,
        OptimizationStrategy.BATCH_PROCESSING
    ])


class PerformanceOptimizer:
    """Main performance optimizer for the evaluation system."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize performance optimizer."""
        self.config = config or OptimizationConfig()
        self.metrics = PerformanceMetrics(
            execution_time=0.0,
            memory_usage=0.0,
            cache_hit_rate=0.0,
            database_queries=0,
            api_calls=0,
            concurrent_operations=0,
            throughput=0.0,
            latency_p95=0.0,
            error_rate=0.0
        )
        self.cache = CachingStrategy(
            backend=self.config.cache_backend,
            ttl=self.config.cache_ttl,
            max_size=self.config.max_cache_size
        )
        self.query_optimizer = QueryOptimizer()
        self._performance_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    async def optimize_evaluation_batch(self, scenarios: List[Scenario], 
                                      evaluation_func: Callable) -> List[EvaluationResult]:
        """Optimize batch evaluation of scenarios."""
        start_time = time.time()
        
        try:
            # Apply optimization strategies
            if OptimizationStrategy.CACHING in self.config.optimization_strategies:
                cached_results = await self._get_cached_results(scenarios)
                uncached_scenarios = [s for s in scenarios if s.id not in cached_results]
            else:
                cached_results = {}
                uncached_scenarios = scenarios
            
            if not uncached_scenarios:
                return list(cached_results.values())
            
            # Apply parallel execution strategy
            if (OptimizationStrategy.PARALLEL_EXECUTION in self.config.optimization_strategies and 
                len(uncached_scenarios) > 1):
                new_results = await self._parallel_evaluation(uncached_scenarios, evaluation_func)
            else:
                new_results = await self._sequential_evaluation(uncached_scenarios, evaluation_func)
            
            # Cache new results
            if OptimizationStrategy.CACHING in self.config.optimization_strategies:
                await self._cache_results(new_results)
            
            # Combine cached and new results
            all_results = list(cached_results.values()) + new_results
            
            # Record performance metrics
            execution_time = time.time() - start_time
            await self._record_performance_metrics(execution_time, len(scenarios), len(cached_results))
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in batch optimization: {e}")
            # Fallback to sequential processing
            return await self._sequential_evaluation(scenarios, evaluation_func)
    
    async def optimize_database_queries(self, query_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize database query execution."""
        return await self.query_optimizer.optimize_queries(query_plans)
    
    async def prefetch_data(self, scenario_ids: List[str]) -> None:
        """Prefetch data for upcoming evaluations."""
        if not self.config.enable_prefetching:
            return
        
        # Prefetch scenarios in batches
        for i in range(0, len(scenario_ids), self.config.prefetch_batch_size):
            batch = scenario_ids[i:i + self.config.prefetch_batch_size]
            asyncio.create_task(self._prefetch_batch(batch))
    
    async def _get_cached_results(self, scenarios: List[Scenario]) -> Dict[str, EvaluationResult]:
        """Get cached evaluation results."""
        cached_results = {}
        
        for scenario in scenarios:
            cache_key = self._generate_cache_key(scenario)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                cached_results[scenario.id] = cached_result
        
        return cached_results
    
    async def _cache_results(self, results: List[EvaluationResult]) -> None:
        """Cache evaluation results."""
        for result in results:
            cache_key = self._generate_cache_key_from_result(result)
            await self.cache.set(cache_key, result)
    
    async def _parallel_evaluation(self, scenarios: List[Scenario], 
                                 evaluation_func: Callable) -> List[EvaluationResult]:
        """Execute evaluations in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def evaluate_with_semaphore(scenario):
            async with semaphore:
                return await evaluation_func(scenario)
        
        tasks = [evaluate_with_semaphore(scenario) for scenario in scenarios]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for scenario {scenarios[i].id}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _sequential_evaluation(self, scenarios: List[Scenario], 
                                   evaluation_func: Callable) -> List[EvaluationResult]:
        """Execute evaluations sequentially."""
        results = []
        
        for scenario in scenarios:
            try:
                result = await evaluation_func(scenario)
                results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for scenario {scenario.id}: {e}")
        
        return results
    
    async def _prefetch_batch(self, scenario_ids: List[str]) -> None:
        """Prefetch a batch of scenarios."""
        try:
            # This would typically load scenarios from database
            logger.debug(f"Prefetching batch of {len(scenario_ids)} scenarios")
            # Implementation would depend on data access layer
        except Exception as e:
            logger.warning(f"Prefetch failed for batch: {e}")
    
    async def _record_performance_metrics(self, execution_time: float, 
                                        total_scenarios: int, cached_count: int) -> None:
        """Record performance metrics."""
        with self._lock:
            self.metrics.execution_time = execution_time
            self.metrics.throughput = total_scenarios / execution_time if execution_time > 0 else 0
            self.metrics.cache_hit_rate = cached_count / total_scenarios if total_scenarios > 0 else 0
            
            # Add to history
            self._performance_history.append({
                'timestamp': time.time(),
                'execution_time': execution_time,
                'scenarios': total_scenarios,
                'cache_hits': cached_count,
                'throughput': self.metrics.throughput
            })
    
    def _generate_cache_key(self, scenario: Scenario) -> str:
        """Generate cache key for scenario evaluation."""
        # Include scenario content and configuration in key
        key_data = f"{scenario.id}:{scenario.title}:{scenario.description}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _generate_cache_key_from_result(self, result: EvaluationResult) -> str:
        """Generate cache key from evaluation result."""
        key_data = f"{result.scenario_id}:{result.agent_provider}:{result.model}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history."""
        with self._lock:
            return list(self._performance_history)


class CachingStrategy:
    """Intelligent caching strategy for evaluation results."""
    
    def __init__(self, backend: CacheBackend = CacheBackend.MEMORY, 
                 ttl: int = 3600, max_size: int = 1000):
        """Initialize caching strategy."""
        self.backend = backend
        self.ttl = ttl
        self.max_size = max_size
        self._memory_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.backend == CacheBackend.MEMORY:
            return await self._memory_get(key)
        elif self.backend == CacheBackend.REDIS:
            return await self._redis_get(key)
        else:
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if self.backend == CacheBackend.MEMORY:
            await self._memory_set(key, value)
        elif self.backend == CacheBackend.REDIS:
            await self._redis_set(key, value)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        if self.backend == CacheBackend.MEMORY:
            await self._memory_delete(key)
        elif self.backend == CacheBackend.REDIS:
            await self._redis_delete(key)
    
    async def clear(self) -> None:
        """Clear all cached values."""
        if self.backend == CacheBackend.MEMORY:
            with self._lock:
                self._memory_cache.clear()
                self._cache_timestamps.clear()
                self._cache_access_counts.clear()
    
    async def _memory_get(self, key: str) -> Optional[Any]:
        """Get from memory cache."""
        with self._lock:
            # Check if key exists and not expired
            if key in self._memory_cache:
                if time.time() - self._cache_timestamps.get(key, 0) < self.ttl:
                    self._cache_access_counts[key] += 1
                    return self._memory_cache[key]
                else:
                    # Expired, remove
                    del self._memory_cache[key]
                    del self._cache_timestamps[key]
                    del self._cache_access_counts[key]
        
        return None
    
    async def _memory_set(self, key: str, value: Any) -> None:
        """Set in memory cache."""
        with self._lock:
            # Check if cache is full
            if len(self._memory_cache) >= self.max_size and key not in self._memory_cache:
                # Evict least recently used item
                lru_key = min(self._cache_access_counts.keys(), 
                            key=lambda k: self._cache_access_counts[k])
                del self._memory_cache[lru_key]
                del self._cache_timestamps[lru_key]
                del self._cache_access_counts[lru_key]
            
            self._memory_cache[key] = value
            self._cache_timestamps[key] = time.time()
            self._cache_access_counts[key] = 1
    
    async def _memory_delete(self, key: str) -> None:
        """Delete from memory cache."""
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                del self._cache_timestamps[key]
                del self._cache_access_counts[key]
    
    async def _redis_get(self, key: str) -> Optional[Any]:
        """Get from Redis cache."""
        try:
            import redis.asyncio as redis
            # This would require Redis configuration
            logger.warning("Redis cache not implemented, falling back to memory cache")
            return await self._memory_get(key)
        except ImportError:
            logger.warning("Redis not available, using memory cache")
            return await self._memory_get(key)
    
    async def _redis_set(self, key: str, value: Any) -> None:
        """Set in Redis cache."""
        try:
            import redis.asyncio as redis
            logger.warning("Redis cache not implemented, falling back to memory cache")
            await self._memory_set(key, value)
        except ImportError:
            await self._memory_set(key, value)
    
    async def _redis_delete(self, key: str) -> None:
        """Delete from Redis cache."""
        await self._memory_delete(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(self._cache_access_counts.values())
            
            return {
                "total_keys": len(self._memory_cache),
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "hit_rate": total_accesses / len(self._cache_access_counts) if self._cache_access_counts else 0,
                "backend": self.backend.value,
                "ttl": self.ttl
            }


class QueryOptimizer:
    """Database query optimization."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_cache: Dict[str, Any] = {}
        self.execution_plans: Dict[str, Dict[str, Any]] = {}
    
    async def optimize_queries(self, query_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize a list of database queries."""
        optimized_plans = []
        
        for plan in query_plans:
            optimized_plan = await self._optimize_single_query(plan)
            optimized_plans.append(optimized_plan)
        
        # Look for opportunities to batch queries
        batched_plans = self._batch_similar_queries(optimized_plans)
        
        return batched_plans
    
    async def _optimize_single_query(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single query."""
        query_type = query_plan.get('type', 'unknown')
        
        # Apply type-specific optimizations
        if query_type == 'select':
            return self._optimize_select_query(query_plan)
        elif query_type == 'insert':
            return self._optimize_insert_query(query_plan)
        elif query_type == 'update':
            return self._optimize_update_query(query_plan)
        else:
            return query_plan
    
    def _optimize_select_query(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize SELECT queries."""
        optimized = query_plan.copy()
        
        # Add index hints if beneficial
        if 'where_conditions' in query_plan:
            optimized['suggested_indexes'] = self._suggest_indexes(query_plan['where_conditions'])
        
        # Optimize ORDER BY
        if 'order_by' in query_plan:
            optimized['order_by_optimization'] = self._optimize_order_by(query_plan['order_by'])
        
        # Add LIMIT if not present and reasonable
        if 'limit' not in query_plan and query_plan.get('expected_rows', 0) > 1000:
            optimized['suggested_limit'] = 1000
        
        return optimized
    
    def _optimize_insert_query(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize INSERT queries."""
        optimized = query_plan.copy()
        
        # Suggest batch inserts
        if query_plan.get('batch_size', 1) == 1:
            optimized['suggested_batch_size'] = 100
        
        return optimized
    
    def _optimize_update_query(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize UPDATE queries."""
        optimized = query_plan.copy()
        
        # Add WHERE clause optimization
        if 'where_conditions' in query_plan:
            optimized['where_optimization'] = self._optimize_where_clause(query_plan['where_conditions'])
        
        return optimized
    
    def _batch_similar_queries(self, query_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch similar queries together."""
        # Group queries by type and table
        query_groups = defaultdict(list)
        
        for plan in query_plans:
            key = f"{plan.get('type')}:{plan.get('table')}"
            query_groups[key].append(plan)
        
        batched_plans = []
        for group_key, plans in query_groups.items():
            if len(plans) > 1 and plans[0].get('type') in ['insert', 'update']:
                # Create batched query
                batched_plan = {
                    'type': 'batch',
                    'original_type': plans[0].get('type'),
                    'table': plans[0].get('table'),
                    'queries': plans,
                    'batch_size': len(plans)
                }
                batched_plans.append(batched_plan)
            else:
                batched_plans.extend(plans)
        
        return batched_plans
    
    def _suggest_indexes(self, where_conditions: List[str]) -> List[str]:
        """Suggest database indexes for WHERE conditions."""
        suggested_indexes = []
        
        for condition in where_conditions:
            # Simple heuristic: suggest index on columns used in WHERE
            if '=' in condition or 'IN' in condition:
                column = condition.split('=')[0].strip() if '=' in condition else condition.split('IN')[0].strip()
                suggested_indexes.append(f"INDEX ON {column}")
        
        return suggested_indexes
    
    def _optimize_order_by(self, order_by: str) -> Dict[str, Any]:
        """Optimize ORDER BY clauses."""
        return {
            'original': order_by,
            'suggestion': f"Consider index on ({order_by}) for better performance"
        }
    
    def _optimize_where_clause(self, where_conditions: List[str]) -> Dict[str, Any]:
        """Optimize WHERE clauses."""
        return {
            'original_conditions': where_conditions,
            'optimization_tips': [
                "Use indexed columns in WHERE conditions",
                "Avoid functions in WHERE clauses",
                "Use LIMIT when appropriate"
            ]
        }


@dataclass
class QuantumState:
    """Represents a quantum-inspired state for optimization."""
    amplitude: complex
    probability: float
    parameters: Dict[str, float]
    
    def __post_init__(self):
        self.probability = abs(self.amplitude) ** 2


class QuantumInspiredOptimizer:
    """Quantum-inspired optimization for skepticism parameter tuning."""
    
    def __init__(self, 
                 population_size: int = 30,
                 max_generations: int = 50,
                 mutation_rate: float = 0.1,
                 quantum_rotation_angle: float = 0.05):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.quantum_rotation_angle = quantum_rotation_angle
        self.population: List[QuantumState] = []
        self.best_solution: Optional[QuantumState] = None
        self.fitness_history: List[float] = []
    
    def initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> None:
        """Initialize quantum population with superposition states."""
        self.population = []
        
        for _ in range(self.population_size):
            # Create quantum superposition of parameter values
            parameters = {}
            amplitude_real = random.uniform(-1, 1)
            amplitude_imag = random.uniform(-1, 1)
            
            for param_name, (min_val, max_val) in parameter_bounds.items():
                parameters[param_name] = random.uniform(min_val, max_val)
            
            # Normalize amplitude
            amplitude = complex(amplitude_real, amplitude_imag)
            norm = abs(amplitude)
            if norm > 0:
                amplitude = amplitude / norm
            
            quantum_state = QuantumState(
                amplitude=amplitude,
                probability=0.0,  # Will be calculated in __post_init__
                parameters=parameters
            )
            
            self.population.append(quantum_state)
    
    def fitness_function(self, 
                        parameters: Dict[str, float], 
                        evaluation_data: List[Tuple[Scenario, SkepticResponse, EvaluationMetrics]]) -> float:
        """Evaluate fitness using quantum-inspired scoring."""
        if not evaluation_data:
            return 0.0
        
        total_fitness = 0.0
        quantum_coherence_bonus = 0.0
        
        for scenario, response, metrics in evaluation_data:
            # Base fitness from evaluation metrics
            base_score = (
                metrics.skepticism_calibration * 0.3 +
                metrics.evidence_standard_score * 0.25 +
                metrics.red_flag_detection * 0.25 +
                metrics.reasoning_quality * 0.2
            )
            
            # Quantum coherence for skepticism alignment
            expected_skepticism = scenario.correct_skepticism_level
            actual_skepticism = 1.0 - response.confidence_level if response.confidence_level <= 0.5 else response.confidence_level
            
            coherence = 1.0 - abs(expected_skepticism - actual_skepticism)
            quantum_coherence_bonus += coherence * 0.1
            
            total_fitness += base_score
        
        # Average fitness with quantum bonuses
        average_fitness = total_fitness / len(evaluation_data)
        coherence_bonus = quantum_coherence_bonus / len(evaluation_data)
        entanglement_bonus = self._calculate_entanglement_bonus(parameters)
        
        return average_fitness + coherence_bonus + entanglement_bonus
    
    def _calculate_entanglement_bonus(self, parameters: Dict[str, float]) -> float:
        """Calculate quantum entanglement bonus for parameter harmony."""
        if len(parameters) < 2:
            return 0.0
        
        param_values = list(parameters.values())
        correlations = []
        
        for i in range(len(param_values)):
            for j in range(i + 1, len(param_values)):
                correlation = abs(param_values[i] * param_values[j])
                correlations.append(correlation)
        
        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            optimal_correlation = 0.5  # Balanced entanglement
            entanglement_score = 1.0 - abs(avg_correlation - optimal_correlation)
            return entanglement_score * 0.05
        
        return 0.0
    
    def quantum_rotation(self, state: QuantumState, target_fitness: float) -> QuantumState:
        """Apply quantum rotation operator for evolution."""
        current_fitness = state.probability
        fitness_diff = target_fitness - current_fitness
        rotation_angle = self.quantum_rotation_angle * fitness_diff
        
        # Quantum rotation transformation
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)
        
        new_amplitude = complex(
            state.amplitude.real * cos_theta - state.amplitude.imag * sin_theta,
            state.amplitude.real * sin_theta + state.amplitude.imag * cos_theta
        )
        
        # Quantum tunneling mutation
        new_parameters = {}
        for param_name, param_value in state.parameters.items():
            if random.random() < self.mutation_rate:
                tunneling_factor = random.gauss(0, 0.1)
                new_parameters[param_name] = param_value + tunneling_factor
            else:
                new_parameters[param_name] = param_value
        
        return QuantumState(
            amplitude=new_amplitude,
            probability=0.0,
            parameters=new_parameters
        )
    
    def optimize(self, 
                parameter_bounds: Dict[str, Tuple[float, float]],
                evaluation_data: List[Tuple[Scenario, SkepticResponse, EvaluationMetrics]]) -> Dict[str, float]:
        """Run quantum-inspired optimization."""
        self.initialize_population(parameter_bounds)
        
        for generation in range(self.max_generations):
            # Evaluate quantum population fitness
            fitness_scores = []
            for state in self.population:
                fitness = self.fitness_function(state.parameters, evaluation_data)
                fitness_scores.append(fitness)
                state.probability = fitness
            
            # Track best solution
            best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if self.best_solution is None or fitness_scores[best_index] > max(self.fitness_history, default=0):
                self.best_solution = self.population[best_index]
            
            self.fitness_history.append(max(fitness_scores))
            
            # Quantum evolution
            target_fitness = max(fitness_scores)
            new_population = []
            
            # Evolve population through quantum operations
            for state in self.population:
                if random.random() < state.probability:
                    evolved_state = self.quantum_rotation(state, target_fitness)
                    new_population.append(evolved_state)
            
            # Quantum crossover for remaining population
            while len(new_population) < self.population_size:
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                
                child_parameters = {}
                for param_name in parameter_bounds.keys():
                    if random.random() < 0.5:
                        child_parameters[param_name] = parent1.parameters[param_name]
                    else:
                        child_parameters[param_name] = parent2.parameters[param_name]
                
                child_amplitude = (parent1.amplitude + parent2.amplitude) / 2
                child_state = QuantumState(
                    amplitude=child_amplitude,
                    probability=0.0,
                    parameters=child_parameters
                )
                new_population.append(child_state)
            
            self.population = new_population
        
        return self.best_solution.parameters if self.best_solution else {}


class SkepticismCalibrator:
    """Advanced skepticism calibration using quantum optimization."""
    
    def __init__(self):
        self.optimizer = QuantumInspiredOptimizer()
        self.calibration_history: List[Dict[str, Any]] = []
    
    def calibrate_agent_parameters(self, 
                                  historical_evaluations: List[Tuple[Scenario, SkepticResponse, EvaluationMetrics]],
                                  target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calibrate agent parameters for optimal skepticism."""
        
        parameter_bounds = {
            "temperature": (0.1, 1.0),
            "skepticism_threshold": (0.3, 0.8),
            "evidence_weight": (0.5, 1.5),
            "confidence_adjustment": (-0.3, 0.3),
            "reasoning_depth": (0.5, 2.0)
        }
        
        # Run quantum optimization
        optimal_params = self.optimizer.optimize(parameter_bounds, historical_evaluations)
        
        # Store calibration results
        calibration_result = {
            "timestamp": time.time(),
            "optimal_parameters": optimal_params,
            "fitness_history": self.optimizer.fitness_history,
            "evaluation_count": len(historical_evaluations),
            "target_metrics": target_metrics or {}
        }
        
        self.calibration_history.append(calibration_result)
        
        return optimal_params
    
    def predict_optimal_skepticism(self, 
                                  scenario: Scenario, 
                                  agent_parameters: Dict[str, float]) -> float:
        """Predict optimal skepticism level using quantum uncertainty."""
        base_skepticism = scenario.correct_skepticism_level
        skepticism_threshold = agent_parameters.get("skepticism_threshold", 0.5)
        evidence_weight = agent_parameters.get("evidence_weight", 1.0)
        
        # Quantum uncertainty calculation
        uncertainty_factor = self._calculate_quantum_uncertainty(scenario)
        
        predicted_skepticism = (
            base_skepticism * 0.4 +
            skepticism_threshold * 0.3 +
            uncertainty_factor * evidence_weight * 0.3
        )
        
        return max(0.0, min(1.0, predicted_skepticism))
    
    def _calculate_quantum_uncertainty(self, scenario: Scenario) -> float:
        """Calculate quantum uncertainty for scenario evaluation."""
        complexity_factors = [
            len(scenario.description) / 1000.0,
            len(scenario.adversary_claim.split()) / 50.0 if hasattr(scenario, 'adversary_claim') else 0.5,
            scenario.metadata.get("evidence_quality", 0.5),
            1.0 - scenario.metadata.get("plausibility", 0.5)
        ]
        
        # Quantum superposition of uncertainty
        uncertainty = sum(factor * random.gauss(1.0, 0.1) for factor in complexity_factors)
        uncertainty = uncertainty / len(complexity_factors)
        
        return max(0.0, min(1.0, uncertainty))
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        if not self.calibration_history:
            return {"error": "No calibration history available"}
        
        latest = self.calibration_history[-1]
        
        return {
            "total_calibrations": len(self.calibration_history),
            "latest_calibration": latest,
            "parameter_evolution": self._analyze_parameter_evolution(),
            "optimization_performance": self._analyze_optimization_performance(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_parameter_evolution(self) -> Dict[str, List[float]]:
        """Analyze parameter evolution across calibrations."""
        evolution = {}
        for calibration in self.calibration_history:
            params = calibration["optimal_parameters"]
            for param_name, param_value in params.items():
                if param_name not in evolution:
                    evolution[param_name] = []
                evolution[param_name].append(param_value)
        return evolution
    
    def _analyze_optimization_performance(self) -> Dict[str, float]:
        """Analyze optimization performance trends."""
        if not self.calibration_history:
            return {}
        
        fitness_trends = [cal["fitness_history"][-1] for cal in self.calibration_history]
        return {
            "average_final_fitness": sum(fitness_trends) / len(fitness_trends),
            "fitness_improvement": fitness_trends[-1] - fitness_trends[0] if len(fitness_trends) > 1 else 0.0,
            "optimization_stability": self._calculate_stability(fitness_trends)
        }
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate optimization stability metric."""
        if len(values) < 2:
            return 1.0
        
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        
        if mean_value > 0:
            relative_variance = variance / (mean_value ** 2)
            stability = 1.0 / (1.0 + relative_variance)
        else:
            stability = 0.0
        
        return stability
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not self.calibration_history:
            return ["No calibration data available"]
        
        latest = self.calibration_history[-1]
        params = latest["optimal_parameters"]
        
        if params.get("temperature", 0.5) > 0.8:
            recommendations.append("High temperature detected - consider lowering for more consistent responses")
        if params.get("skepticism_threshold", 0.5) < 0.3:
            recommendations.append("Low skepticism threshold - agent may be too trusting")
        if len(latest["fitness_history"]) > 10 and latest["fitness_history"][-1] < 0.7:
            recommendations.append("Suboptimal fitness achieved - consider expanding parameter search space")
        
        return recommendations if recommendations else ["Optimization appears well-calibrated"]


class QuantumInspiredOptimizer:
    """Advanced quantum-inspired optimization with quantum annealing and superposition."""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        """Initialize quantum optimizer."""
        self.population_size = population_size
        self.generations = generations
        self.quantum_states: List[QuantumOptimizationState] = []
        self.coherence_threshold = 0.8
        self.entanglement_strength = 0.5
        self.annealing_schedule = self._create_annealing_schedule()
        
        # Quantum gates for parameter manipulation
        self.rotation_gates = self._initialize_rotation_gates()
        self.superposition_cache = {}
        
    def _create_annealing_schedule(self) -> List[float]:
        """Create quantum annealing temperature schedule."""
        initial_temp = 10.0
        final_temp = 0.01
        return [
            initial_temp * ((final_temp / initial_temp) ** (gen / self.generations))
            for gen in range(self.generations)
        ]
    
    def _initialize_rotation_gates(self) -> Dict[str, Callable[[float, float], float]]:
        """Initialize quantum rotation gates for parameter evolution."""
        return {
            'pauli_x': lambda theta, param: math.cos(theta) * param + math.sin(theta) * (1 - param),
            'pauli_y': lambda theta, param: param * math.cos(theta) + (1 - param) * math.sin(theta),
            'pauli_z': lambda theta, param: param * math.cos(2 * theta),
            'hadamard': lambda theta, param: (param + (1 - param)) / math.sqrt(2)
        }
    
    async def optimize_parameters(self, 
                                objective_func: Callable[[Dict[str, float]], float],
                                parameter_bounds: Dict[str, Tuple[float, float]],
                                target_fitness: float = 0.9) -> Dict[str, float]:
        """Optimize parameters using quantum-inspired genetic algorithm."""
        
        # Initialize quantum population
        await self._initialize_quantum_population(parameter_bounds)
        
        best_fitness = 0.0
        best_parameters = {}
        optimization_history = []
        
        for generation in range(self.generations):
            current_temp = self.annealing_schedule[generation]
            
            # Quantum superposition evaluation
            fitness_scores = await self._evaluate_population_in_superposition(objective_func)
            
            # Find best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_parameters = self.quantum_states[gen_best_idx].parameters.copy()
            
            # Record progress
            optimization_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'average_fitness': np.mean(fitness_scores),
                'quantum_coherence': self._measure_population_coherence(),
                'temperature': current_temp
            })
            
            # Early stopping if target reached
            if best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at generation {generation}")
                break
            
            # Quantum evolution operations
            await self._quantum_evolution_step(fitness_scores, current_temp)
            
            # Maintain quantum coherence
            await self._maintain_coherence()
        
        # Final quantum measurement
        final_state = await self._collapse_to_best_state(best_parameters)
        
        return {
            'parameters': best_parameters,
            'fitness': best_fitness,
            'generations': generation + 1,
            'final_coherence': final_state.coherence_level,
            'optimization_history': optimization_history
        }
    
    async def _initialize_quantum_population(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """Initialize population with quantum superposition states."""
        self.quantum_states = []
        
        for _ in range(self.population_size):
            # Create random parameters within bounds
            parameters = {}
            for param_name, (min_val, max_val) in parameter_bounds.items():
                parameters[param_name] = random.uniform(min_val, max_val)
            
            # Create quantum amplitude (complex number)
            amplitude = complex(
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            )
            # Normalize amplitude
            magnitude = abs(amplitude)
            if magnitude > 0:
                amplitude = amplitude / magnitude
            
            state = QuantumOptimizationState(
                amplitude=amplitude,
                parameters=parameters,
                fitness_score=0.0,
                coherence_level=1.0,
                entanglement_measures={}
            )
            
            self.quantum_states.append(state)
    
    async def _evaluate_population_in_superposition(self, 
                                                  objective_func: Callable[[Dict[str, float]], float]) -> List[float]:
        """Evaluate population using quantum superposition principles."""
        fitness_scores = []
        
        # Create evaluation tasks
        tasks = []
        for state in self.quantum_states:
            # Weight evaluation by probability amplitude
            probability = abs(state.amplitude) ** 2
            tasks.append(self._quantum_evaluate(objective_func, state, probability))
        
        # Execute in parallel with quantum interference
        results = await asyncio.gather(*tasks)
        
        for i, (fitness, quantum_effects) in enumerate(results):
            self.quantum_states[i].fitness_score = fitness
            self.quantum_states[i].entanglement_measures = quantum_effects
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    async def _quantum_evaluate(self, 
                              objective_func: Callable, 
                              state: QuantumOptimizationState, 
                              probability: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate single quantum state with quantum effects."""
        try:
            # Base fitness evaluation
            base_fitness = await asyncio.to_thread(objective_func, state.parameters)
            
            # Apply quantum effects
            quantum_interference = self._calculate_quantum_interference(state)
            quantum_tunneling = self._apply_quantum_tunneling(base_fitness, probability)
            
            # Final fitness with quantum corrections
            final_fitness = base_fitness * (1 + quantum_interference) + quantum_tunneling
            final_fitness = max(0.0, min(1.0, final_fitness))  # Bound to [0,1]
            
            quantum_effects = {
                'interference': quantum_interference,
                'tunneling': quantum_tunneling,
                'probability': probability
            }
            
            return final_fitness, quantum_effects
            
        except Exception as e:
            logger.error(f"Quantum evaluation error: {e}")
            return 0.0, {}
    
    def _calculate_quantum_interference(self, state: QuantumOptimizationState) -> float:
        """Calculate quantum interference effects."""
        # Interference based on phase relationships with other states
        interference = 0.0
        
        for other_state in self.quantum_states:
            if other_state == state:
                continue
            
            # Phase difference
            phase_diff = np.angle(state.amplitude) - np.angle(other_state.amplitude)
            
            # Parameter similarity
            param_similarity = self._calculate_parameter_similarity(
                state.parameters, other_state.parameters
            )
            
            # Constructive/destructive interference
            interference_effect = param_similarity * math.cos(phase_diff)
            interference += interference_effect * 0.1  # Scale factor
        
        return interference / len(self.quantum_states) if self.quantum_states else 0.0
    
    def _apply_quantum_tunneling(self, fitness: float, probability: float) -> float:
        """Apply quantum tunneling to escape local optima."""
        if fitness < 0.5:  # Low fitness - tunneling more likely
            tunneling_probability = (1 - fitness) * probability * 0.2
            if random.random() < tunneling_probability:
                # Tunnel to higher fitness region
                tunnel_boost = random.uniform(0.1, 0.3)
                return tunnel_boost
        
        return 0.0
    
    async def _quantum_evolution_step(self, fitness_scores: List[float], temperature: float):
        """Perform quantum evolution step."""
        
        # Selection based on fitness and probability amplitude
        selected_indices = self._quantum_selection(fitness_scores)
        
        # Quantum crossover with entanglement
        await self._quantum_crossover(selected_indices)
        
        # Quantum mutation with controlled decoherence
        await self._quantum_mutation(temperature)
        
        # Update quantum phases
        self._update_quantum_phases()
    
    def _quantum_selection(self, fitness_scores: List[float]) -> List[int]:
        """Select states for evolution using quantum selection pressure."""
        selection_probs = []
        
        for i, (fitness, state) in enumerate(zip(fitness_scores, self.quantum_states)):
            # Combine fitness and quantum probability
            quantum_prob = abs(state.amplitude) ** 2
            combined_prob = fitness * 0.7 + quantum_prob * 0.3
            selection_probs.append(combined_prob)
        
        # Normalize probabilities
        total_prob = sum(selection_probs)
        if total_prob > 0:
            selection_probs = [p / total_prob for p in selection_probs]
        else:
            selection_probs = [1.0 / len(selection_probs)] * len(selection_probs)
        
        # Select indices based on quantum probabilities
        selected = []
        for _ in range(len(self.quantum_states) // 2):
            idx = np.random.choice(len(selection_probs), p=selection_probs)
            selected.append(idx)
        
        return selected
    
    async def _quantum_crossover(self, selected_indices: List[int]):
        """Perform quantum crossover with entanglement."""
        new_states = []
        
        for i in range(0, len(selected_indices) - 1, 2):
            parent1 = self.quantum_states[selected_indices[i]]
            parent2 = self.quantum_states[selected_indices[i + 1]]
            
            # Create entangled offspring
            child1, child2 = await self._create_entangled_offspring(parent1, parent2)
            new_states.extend([child1, child2])
        
        # Replace worst performing states
        fitness_scores = [state.fitness_score for state in self.quantum_states]
        worst_indices = np.argsort(fitness_scores)[:len(new_states)]
        
        for i, new_state in enumerate(new_states):
            if i < len(worst_indices):
                self.quantum_states[worst_indices[i]] = new_state
    
    async def _create_entangled_offspring(self, 
                                        parent1: QuantumOptimizationState, 
                                        parent2: QuantumOptimizationState) -> Tuple[QuantumOptimizationState, QuantumOptimizationState]:
        """Create quantum entangled offspring."""
        
        # Quantum entanglement in parameters
        child1_params = {}
        child2_params = {}
        
        for param_name in parent1.parameters:
            val1 = parent1.parameters[param_name]
            val2 = parent2.parameters[param_name]
            
            # Quantum crossover with controlled entanglement
            entanglement_factor = random.uniform(0, self.entanglement_strength)
            
            # Entangled parameter values
            child1_params[param_name] = val1 * (1 - entanglement_factor) + val2 * entanglement_factor
            child2_params[param_name] = val2 * (1 - entanglement_factor) + val1 * entanglement_factor
        
        # Entangled amplitudes
        combined_amplitude = (parent1.amplitude + parent2.amplitude) / 2
        
        child1 = QuantumOptimizationState(
            amplitude=combined_amplitude,
            parameters=child1_params,
            fitness_score=0.0,
            coherence_level=(parent1.coherence_level + parent2.coherence_level) / 2,
            entanglement_measures={'entangled_with': 'child2', 'entanglement_factor': entanglement_factor}
        )
        
        child2 = QuantumOptimizationState(
            amplitude=combined_amplitude.conjugate(),
            parameters=child2_params,
            fitness_score=0.0,
            coherence_level=(parent1.coherence_level + parent2.coherence_level) / 2,
            entanglement_measures={'entangled_with': 'child1', 'entanglement_factor': entanglement_factor}
        )
        
        return child1, child2
    
    async def _quantum_mutation(self, temperature: float):
        """Apply quantum mutation with controlled decoherence."""
        for state in self.quantum_states:
            # Mutation probability based on temperature (annealing)
            mutation_prob = temperature / 10.0  # Scale temperature
            
            if random.random() < mutation_prob:
                # Apply quantum gates for parameter mutation
                gate_type = random.choice(list(self.rotation_gates.keys()))
                gate_func = self.rotation_gates[gate_type]
                
                # Random rotation angle
                theta = random.uniform(-math.pi, math.pi)
                
                # Mutate parameters using quantum gate
                for param_name in state.parameters:
                    old_val = state.parameters[param_name]
                    new_val = gate_func(theta, old_val)
                    
                    # Ensure parameter bounds (assuming [0,1] normalized)
                    state.parameters[param_name] = max(0.0, min(1.0, new_val))
                
                # Update amplitude with mutation
                mutation_amplitude = complex(
                    random.gauss(0, 0.1),
                    random.gauss(0, 0.1)
                )
                state.amplitude += mutation_amplitude
                
                # Normalize amplitude
                magnitude = abs(state.amplitude)
                if magnitude > 0:
                    state.amplitude = state.amplitude / magnitude
                
                # Reduce coherence due to decoherence
                state.coherence_level *= 0.95
    
    def _update_quantum_phases(self):
        """Update quantum phases based on interactions."""
        for i, state in enumerate(self.quantum_states):
            # Phase evolution based on fitness
            phase_shift = state.fitness_score * 0.1
            
            # Apply phase shift
            current_phase = np.angle(state.amplitude)
            new_phase = current_phase + phase_shift
            magnitude = abs(state.amplitude)
            
            state.amplitude = magnitude * complex(math.cos(new_phase), math.sin(new_phase))
    
    async def _maintain_coherence(self):
        """Maintain quantum coherence across population."""
        avg_coherence = sum(state.coherence_level for state in self.quantum_states) / len(self.quantum_states)
        
        if avg_coherence < self.coherence_threshold:
            # Apply coherence restoration
            for state in self.quantum_states:
                if state.coherence_level < self.coherence_threshold:
                    # Restore coherence by normalizing amplitude
                    magnitude = abs(state.amplitude)
                    if magnitude > 0:
                        state.amplitude = state.amplitude / magnitude
                    
                    state.coherence_level = min(1.0, state.coherence_level + 0.1)
    
    def _measure_population_coherence(self) -> float:
        """Measure overall population coherence."""
        coherence_values = [state.coherence_level for state in self.quantum_states]
        return sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
    
    async def _collapse_to_best_state(self, best_parameters: Dict[str, float]) -> QuantumOptimizationState:
        """Collapse quantum superposition to best measurement."""
        # Find state with best parameters
        best_state = None
        min_distance = float('inf')
        
        for state in self.quantum_states:
            distance = self._calculate_parameter_distance(state.parameters, best_parameters)
            if distance < min_distance:
                min_distance = distance
                best_state = state
        
        if best_state:
            # Collapse superposition
            best_state.amplitude = complex(1.0, 0.0)  # Pure state
            best_state.coherence_level = 1.0
            
        return best_state or self.quantum_states[0]
    
    def _calculate_parameter_similarity(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """Calculate similarity between parameter sets."""
        if not params1 or not params2:
            return 0.0
        
        distances = []
        for key in params1:
            if key in params2:
                distances.append(abs(params1[key] - params2[key]))
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            return 1.0 - avg_distance  # Similarity = 1 - distance
        
        return 0.0
    
    def _calculate_parameter_distance(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between parameter sets."""
        if not params1 or not params2:
            return float('inf')
        
        squared_diffs = []
        for key in params1:
            if key in params2:
                squared_diffs.append((params1[key] - params2[key]) ** 2)
        
        return math.sqrt(sum(squared_diffs)) if squared_diffs else float('inf')


class FederatedLearningOptimizer:
    """Federated learning system for distributed agent optimization."""
    
    def __init__(self, node_id: str, max_nodes: int = 10):
        """Initialize federated learning node."""
        self.node_id = node_id
        self.max_nodes = max_nodes
        self.local_model = {}
        self.global_model = {}
        self.node_contributions = {}
        self.communication_rounds = 0
        
        # Privacy and security settings
        self.differential_privacy_epsilon = 1.0
        self.secure_aggregation = True
        self.min_nodes_for_update = 3
        
        # Performance tracking
        self.federation_history = []
        self.node_performance_metrics = {}
    
    async def federated_optimization_round(self, 
                                         local_data: List[EvaluationResult],
                                         global_parameters: Dict[str, float]) -> Dict[str, float]:
        """Execute one round of federated learning optimization."""
        
        # Local training phase
        local_update = await self._local_training_phase(local_data, global_parameters)
        
        # Privacy-preserving aggregation
        if self.differential_privacy_epsilon > 0:
            local_update = self._apply_differential_privacy(local_update)
        
        # Prepare for aggregation
        aggregation_data = {
            'node_id': self.node_id,
            'parameters': local_update,
            'data_size': len(local_data),
            'local_performance': self._evaluate_local_performance(local_data),
            'round': self.communication_rounds
        }
        
        return aggregation_data
    
    async def _local_training_phase(self, 
                                  local_data: List[EvaluationResult],
                                  global_parameters: Dict[str, float]) -> Dict[str, float]:
        """Train local model on node's data."""
        
        # Initialize local model from global parameters
        self.local_model = global_parameters.copy()
        
        # Extract local optimization objectives
        local_objectives = self._extract_local_objectives(local_data)
        
        # Local optimization using quantum-inspired methods
        quantum_optimizer = QuantumInspiredOptimizer(population_size=20, generations=50)
        
        # Define local objective function
        async def local_objective(params: Dict[str, float]) -> float:
            return await self._evaluate_local_objective(params, local_objectives)
        
        # Optimize locally
        optimization_result = await quantum_optimizer.optimize_parameters(
            objective_func=local_objective,
            parameter_bounds={key: (0.0, 1.0) for key in global_parameters.keys()},
            target_fitness=0.85
        )
        
        return optimization_result['parameters']
    
    def _extract_local_objectives(self, local_data: List[EvaluationResult]) -> Dict[str, Any]:
        """Extract optimization objectives from local evaluation data."""
        objectives = {
            'accuracy_targets': [],
            'performance_constraints': {},
            'domain_specific_metrics': {}
        }
        
        for result in local_data:
            # Extract target metrics for optimization
            objectives['accuracy_targets'].append({
                'scenario_category': result.scenario.category.value,
                'target_skepticism': result.scenario.correct_skepticism_level,
                'achieved_metrics': {
                    'skepticism_calibration': result.metrics.skepticism_calibration,
                    'evidence_standard': result.metrics.evidence_standard_score,
                    'red_flag_detection': result.metrics.red_flag_detection
                }
            })
        
        return objectives
    
    async def _evaluate_local_objective(self, 
                                      params: Dict[str, float],
                                      objectives: Dict[str, Any]) -> float:
        """Evaluate parameters against local objectives."""
        
        total_score = 0.0
        count = 0
        
        for target in objectives['accuracy_targets']:
            # Simulate parameter performance against target
            predicted_performance = self._predict_performance(params, target)
            
            # Calculate alignment with target
            target_skepticism = target['target_skepticism']
            achieved_metrics = target['achieved_metrics']
            
            # Weighted scoring
            score = (
                achieved_metrics['skepticism_calibration'] * 0.4 +
                achieved_metrics['evidence_standard'] * 0.3 +
                achieved_metrics['red_flag_detection'] * 0.3
            )
            
            # Adjust for parameter influence
            parameter_influence = sum(params.values()) / len(params)
            adjusted_score = score * (0.5 + 0.5 * parameter_influence)
            
            total_score += adjusted_score
            count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _predict_performance(self, 
                           params: Dict[str, float], 
                           target: Dict[str, Any]) -> float:
        """Predict performance of parameters on target scenario."""
        
        # Simple heuristic prediction model
        # In practice, this would use learned models
        
        temperature = params.get('temperature', 0.5)
        confidence_threshold = params.get('confidence_threshold', 0.5)
        skepticism_bias = params.get('skepticism_bias', 0.5)
        
        target_skepticism = target['target_skepticism']
        
        # Predict skepticism alignment
        predicted_skepticism = (
            skepticism_bias * 0.5 +
            (1 - temperature) * 0.3 +
            confidence_threshold * 0.2
        )
        
        # Calculate alignment score
        alignment = 1.0 - abs(predicted_skepticism - target_skepticism)
        
        return max(0.0, alignment)
    
    def _apply_differential_privacy(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply differential privacy to protect local data."""
        
        epsilon = self.differential_privacy_epsilon
        sensitivity = 0.1  # Assume bounded parameter sensitivity
        
        # Add Laplace noise for differential privacy
        private_params = {}
        for key, value in parameters.items():
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(0, noise_scale)
            private_params[key] = max(0.0, min(1.0, value + noise))
        
        return private_params
    
    async def aggregate_global_model(self, 
                                   node_contributions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate contributions from multiple nodes into global model."""
        
        if len(node_contributions) < self.min_nodes_for_update:
            logger.warning(f"Insufficient nodes for update: {len(node_contributions)} < {self.min_nodes_for_update}")
            return self.global_model
        
        # Weighted federated averaging
        aggregated_params = {}
        total_weight = 0.0
        
        for contribution in node_contributions:
            node_weight = self._calculate_node_weight(contribution)
            
            for param_name, param_value in contribution['parameters'].items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = 0.0
                
                aggregated_params[param_name] += param_value * node_weight
            
            total_weight += node_weight
        
        # Normalize by total weight
        if total_weight > 0:
            for param_name in aggregated_params:
                aggregated_params[param_name] /= total_weight
        
        # Update global model
        self.global_model = aggregated_params
        self.communication_rounds += 1
        
        # Record federation round
        self.federation_history.append({
            'round': self.communication_rounds,
            'participating_nodes': len(node_contributions),
            'global_parameters': self.global_model.copy(),
            'aggregation_weights': {
                contrib['node_id']: self._calculate_node_weight(contrib)
                for contrib in node_contributions
            }
        })
        
        return self.global_model
    
    def _calculate_node_weight(self, contribution: Dict[str, Any]) -> float:
        """Calculate weight for node contribution in aggregation."""
        
        # Base weight on data size
        data_size_weight = contribution['data_size'] / 100.0  # Normalize
        
        # Weight by local performance
        performance_weight = contribution['local_performance']
        
        # Weight by node reliability (from history)
        reliability_weight = self._get_node_reliability(contribution['node_id'])
        
        # Combined weight
        total_weight = (
            data_size_weight * 0.4 +
            performance_weight * 0.4 +
            reliability_weight * 0.2
        )
        
        return max(0.1, min(2.0, total_weight))  # Bound weights
    
    def _get_node_reliability(self, node_id: str) -> float:
        """Get reliability score for a node."""
        if node_id not in self.node_performance_metrics:
            return 1.0  # Default reliability for new nodes
        
        metrics = self.node_performance_metrics[node_id]
        
        # Calculate reliability based on historical performance
        consistency = metrics.get('consistency', 1.0)
        participation = metrics.get('participation_rate', 1.0)
        quality = metrics.get('contribution_quality', 1.0)
        
        return (consistency + participation + quality) / 3.0
    
    def _evaluate_local_performance(self, local_data: List[EvaluationResult]) -> float:
        """Evaluate performance of local optimization."""
        if not local_data:
            return 0.0
        
        # Calculate average performance metrics
        total_score = 0.0
        for result in local_data:
            total_score += result.metrics.overall_score
        
        return total_score / len(local_data)
    
    def update_node_performance_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update performance metrics for a node."""
        if node_id not in self.node_performance_metrics:
            self.node_performance_metrics[node_id] = {}
        
        self.node_performance_metrics[node_id].update(metrics)
    
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get summary of federated learning progress."""
        if not self.federation_history:
            return {'status': 'No federation rounds completed'}
        
        latest_round = self.federation_history[-1]
        
        return {
            'total_rounds': self.communication_rounds,
            'participating_nodes': latest_round['participating_nodes'],
            'global_parameters': self.global_model,
            'node_count': len(self.node_performance_metrics),
            'average_node_reliability': sum(
                self._get_node_reliability(node_id) 
                for node_id in self.node_performance_metrics
            ) / len(self.node_performance_metrics) if self.node_performance_metrics else 0.0,
            'federation_convergence': self._measure_convergence()
        }
    
    def _measure_convergence(self) -> float:
        """Measure convergence of federated learning."""
        if len(self.federation_history) < 2:
            return 0.0
        
        # Compare parameter changes between rounds
        prev_params = self.federation_history[-2]['global_parameters']
        current_params = self.federation_history[-1]['global_parameters']
        
        if not prev_params or not current_params:
            return 0.0
        
        # Calculate parameter stability
        total_change = 0.0
        param_count = 0
        
        for key in current_params:
            if key in prev_params:
                change = abs(current_params[key] - prev_params[key])
                total_change += change
                param_count += 1
        
        if param_count > 0:
            avg_change = total_change / param_count
            convergence = max(0.0, 1.0 - avg_change)  # Lower change = higher convergence
            return convergence
        
        return 0.0