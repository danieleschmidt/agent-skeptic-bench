"""Performance optimization algorithms for Agent Skeptic Bench with quantum-inspired enhancements."""

import logging
import asyncio
import time
import math
import random
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

from ..models import EvaluationResult, Scenario, EvaluationMetrics, SkepticResponse


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
    parameters: Dict[str, float]
    probability: float = 0.0
    
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