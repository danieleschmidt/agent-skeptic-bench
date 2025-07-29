# Performance Optimization Guide

Optimization strategies and benchmarking for Agent Skeptic Bench at scale.

## Performance Architecture

### System Requirements

| Deployment Scale | CPU Cores | Memory (GB) | Storage (GB) | Network (Mbps) |
|------------------|-----------|-------------|--------------|----------------|
| **Development**  | 2         | 4           | 20           | 10             |
| **Small Production** | 4     | 8           | 50           | 50             |
| **Medium Production** | 8    | 16          | 100          | 100            |
| **Large Production** | 16+   | 32+         | 500+         | 1000+          |

### Bottleneck Analysis

```python
# src/agent_skeptic_bench/profiling.py
import cProfile
import pstats
import io
from functools import wraps
from typing import Callable, Any
import time
import psutil
import logging

class PerformanceProfiler:
    """Comprehensive performance profiling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'execution_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'api_call_times': []
        }
    
    def profile_function(self, func: Callable) -> Callable:
        """Profile function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss
            cpu_before = process.cpu_percent()
            
            # Profile execution
            profiler = cProfile.Profile()
            profiler.enable()
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                profiler.disable()
            
            # Memory after
            memory_after = process.memory_info().rss
            cpu_after = process.cpu_percent()
            
            # Record metrics
            execution_time = end_time - start_time
            memory_delta = memory_after - memory_before
            cpu_delta = cpu_after - cpu_before
            
            self.metrics['execution_times'].append(execution_time)
            self.metrics['memory_usage'].append(memory_delta)
            self.metrics['cpu_usage'].append(cpu_delta)
            
            # Log performance data
            self.logger.info(
                f"Performance: {func.__name__} took {execution_time:.3f}s, "
                f"memory delta: {memory_delta / 1024 / 1024:.2f}MB, "
                f"CPU delta: {cpu_delta:.1f}%"
            )
            
            # Generate profile report if execution is slow
            if execution_time > 5.0:  # 5 second threshold
                self._generate_profile_report(profiler, func.__name__)
            
            return result
        
        return wrapper
    
    def _generate_profile_report(self, profiler: cProfile.Profile, func_name: str):
        """Generate detailed profile report for slow functions."""
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        self.logger.warning(f"Slow execution profile for {func_name}:\n{profile_output}")
    
    def get_performance_summary(self) -> dict:
        """Get performance metrics summary."""
        if not self.metrics['execution_times']:
            return {}
        
        import statistics
        
        return {
            'avg_execution_time': statistics.mean(self.metrics['execution_times']),
            'p95_execution_time': statistics.quantiles(self.metrics['execution_times'], n=20)[18],
            'avg_memory_usage': statistics.mean(self.metrics['memory_usage']),
            'max_memory_usage': max(self.metrics['memory_usage']),
            'avg_cpu_usage': statistics.mean(self.metrics['cpu_usage']),
            'total_calls': len(self.metrics['execution_times'])
        }

# Global profiler instance
profiler = PerformanceProfiler()
```

## Concurrent Processing

### Asynchronous Evaluation

```python
# src/agent_skeptic_bench/async_evaluation.py
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

@dataclass
class EvaluationTask:
    """Individual evaluation task."""
    scenario_id: str
    model: str
    category: str
    priority: int = 1

class AsyncEvaluationEngine:
    """High-performance asynchronous evaluation engine."""
    
    def __init__(self, 
                 max_concurrent_evaluations: int = 10,
                 max_api_requests_per_second: int = 20,
                 timeout_seconds: int = 60):
        self.max_concurrent = max_concurrent_evaluations
        self.rate_limit = max_api_requests_per_second
        self.timeout = timeout_seconds
        self.logger = logging.getLogger(__name__)
        
        # Create semaphores for concurrency control
        self.evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)
        self.api_semaphore = asyncio.Semaphore(max_api_requests_per_second)
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def evaluate_batch(self, tasks: List[EvaluationTask]) -> List[Dict[str, Any]]:
        """Evaluate multiple scenarios concurrently."""
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)
        
        # Create coroutines for all tasks
        coroutines = [self._evaluate_single(task) for task in sorted_tasks]
        
        # Execute with progress tracking
        results = []
        completed = 0
        total = len(coroutines)
        
        for coro in asyncio.as_completed(coroutines):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                if completed % 10 == 0:  # Log progress every 10 completions
                    self.logger.info(f"Completed {completed}/{total} evaluations")
                    
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                results.append({'error': str(e)})
        
        return results
    
    async def _evaluate_single(self, task: EvaluationTask) -> Dict[str, Any]:
        """Evaluate a single scenario with concurrency control."""
        async with self.evaluation_semaphore:
            try:
                # Load scenario (cached)
                scenario = await self._load_scenario(task.scenario_id)
                
                # Create agent
                agent = await self._create_agent(task.model)
                
                # Run evaluation with timeout
                result = await asyncio.wait_for(
                    self._run_evaluation(agent, scenario),
                    timeout=self.timeout
                )
                
                return {
                    'scenario_id': task.scenario_id,
                    'model': task.model,
                    'category': task.category,
                    'result': result,
                    'status': 'success'
                }
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Evaluation timeout for {task.scenario_id}")
                return {
                    'scenario_id': task.scenario_id,
                    'model': task.model,
                    'status': 'timeout'
                }
            except Exception as e:
                self.logger.error(f"Evaluation error for {task.scenario_id}: {e}")
                return {
                    'scenario_id': task.scenario_id,
                    'model': task.model,
                    'status': 'error',
                    'error': str(e)
                }
    
    async def _run_evaluation(self, agent, scenario) -> Dict[str, Any]:
        """Run the actual evaluation with API rate limiting."""
        async with self.api_semaphore:
            # Simulate API call delay for rate limiting
            await asyncio.sleep(1.0 / self.rate_limit)
            
            # Run CPU-intensive evaluation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._cpu_intensive_evaluation,
                agent,
                scenario
            )
            
            return result
    
    def _cpu_intensive_evaluation(self, agent, scenario) -> Dict[str, Any]:
        """CPU-intensive evaluation logic (runs in thread pool)."""
        # This would contain the actual evaluation logic
        # that doesn't involve async API calls
        import time
        time.sleep(0.1)  # Simulate processing
        
        return {
            'skepticism_score': 0.75,
            'evidence_requests': ['peer_review', 'replication'],
            'final_belief': 0.25
        }
    
    async def _load_scenario(self, scenario_id: str):
        """Load scenario with caching."""
        # Implement scenario caching here
        return {'id': scenario_id, 'content': 'mock scenario'}
    
    async def _create_agent(self, model: str):
        """Create agent with connection pooling."""
        # Implement connection pooling here
        return {'model': model, 'connection': 'pooled'}
```

### Connection Pooling

```python
# src/agent_skeptic_bench/connection_pool.py
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

class APIConnectionPool:
    """Optimized connection pooling for AI APIs."""
    
    def __init__(self):
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.logger = logging.getLogger(__name__)
        
        # Connection pool settings optimized for AI APIs
        self.connector_config = {
            'limit': 100,  # Total connection pool size
            'limit_per_host': 20,  # Per-host limit
            'ttl_dns_cache': 300,  # DNS cache TTL
            'use_dns_cache': True,
            'keepalive_timeout': 60,  # Keep connections alive
            'enable_cleanup_closed': True
        }
        
        # Timeout settings for different API providers
        self.timeout_configs = {
            'openai': aiohttp.ClientTimeout(
                total=120,  # Total timeout
                connect=10,  # Connection timeout
                sock_read=60  # Socket read timeout
            ),
            'anthropic': aiohttp.ClientTimeout(
                total=180,  # Anthropic can be slower
                connect=10,
                sock_read=90
            ),
            'google': aiohttp.ClientTimeout(
                total=120,
                connect=10,
                sock_read=60
            )
        }
    
    async def get_session(self, provider: str) -> aiohttp.ClientSession:
        """Get optimized session for API provider."""
        if provider not in self.sessions:
            connector = aiohttp.TCPConnector(**self.connector_config)
            timeout = self.timeout_configs.get(provider, aiohttp.ClientTimeout(total=60))
            
            self.sessions[provider] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Agent-Skeptic-Bench/1.0.0',
                    'Connection': 'keep-alive'
                }
            )
            
            self.logger.info(f"Created optimized session for {provider}")
        
        return self.sessions[provider]
    
    async def close_all(self):
        """Clean shutdown of all sessions."""
        for provider, session in self.sessions.items():
            await session.close()
            self.logger.info(f"Closed session for {provider}")
        
        self.sessions.clear()
    
    @asynccontextmanager
    async def request(self, provider: str, method: str, url: str, **kwargs):
        """Make HTTP request with connection pooling."""
        session = await self.get_session(provider)
        
        try:
            async with session.request(method, url, **kwargs) as response:
                yield response
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error for {provider}: {e}")
            raise
        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout error for {provider}: {e}")
            raise

# Global connection pool
connection_pool = APIConnectionPool()
```

## Caching Strategies

### Multi-Level Caching

```python
# src/agent_skeptic_bench/caching.py
import redis
import json
import hashlib
from typing import Any, Optional, Dict, Union
from datetime import timedelta
import logging
from functools import wraps

class CacheManager:
    """Multi-level caching with Redis and in-memory tiers."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 max_memory_cache_size: int = 1000):
        # Redis for persistent, distributed caching
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # In-memory cache for frequently accessed data
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_order = []  # LRU tracking
        self.max_memory_size = max_memory_cache_size
        
        self.logger = logging.getLogger(__name__)
        
        # Cache TTL settings by data type
        self.ttl_config = {
            'scenario': timedelta(hours=24),
            'model_response': timedelta(hours=1),
            'evaluation_result': timedelta(days=7),
            'agent_config': timedelta(hours=12)
        }
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def _manage_memory_cache(self, key: str, value: Any):
        """Manage in-memory cache size (LRU eviction)."""
        if key in self.memory_cache:
            # Move to end (most recently used)
            self.memory_cache_order.remove(key)
        elif len(self.memory_cache) >= self.max_memory_size:
            # Evict least recently used
            oldest_key = self.memory_cache_order.pop(0)
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        self.memory_cache_order.append(key)
    
    async def get(self, cache_type: str, **kwargs) -> Optional[Any]:
        """Get cached value with multi-level lookup."""
        key = self._generate_key(cache_type, **kwargs)
        
        # Level 1: Memory cache (fastest)
        if key in self.memory_cache:
            # Update LRU order
            self.memory_cache_order.remove(key)
            self.memory_cache_order.append(key)
            self.logger.debug(f"Memory cache hit: {key}")
            return self.memory_cache[key]
        
        # Level 2: Redis cache
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                value = json.loads(cached_data)
                # Populate memory cache
                self._manage_memory_cache(key, value)
                self.logger.debug(f"Redis cache hit: {key}")
                return value
        except (json.JSONDecodeError, redis.RedisError) as e:
            self.logger.error(f"Cache retrieval error for {key}: {e}")
        
        self.logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, cache_type: str, value: Any, **kwargs):
        """Set cached value in both levels."""
        key = self._generate_key(cache_type, **kwargs)
        ttl = self.ttl_config.get(cache_type, timedelta(hours=1))
        
        # Store in memory cache
        self._manage_memory_cache(key, value)
        
        # Store in Redis with TTL
        try:
            serialized_value = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized_value)
            self.logger.debug(f"Cached value: {key} (TTL: {ttl})")
        except (json.JSONEncodeError, redis.RedisError) as e:
            self.logger.error(f"Cache storage error for {key}: {e}")
    
    async def invalidate(self, cache_type: str, **kwargs):
        """Invalidate cached value."""
        key = self._generate_key(cache_type, **kwargs)
        
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            self.memory_cache_order.remove(key)
        
        # Remove from Redis
        try:
            self.redis_client.delete(key)
            self.logger.debug(f"Invalidated cache: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Cache invalidation error for {key}: {e}")
    
    def cached(self, cache_type: str, **cache_kwargs):
        """Decorator for automatic caching."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Merge cache kwargs with function kwargs for key generation
                cache_key_params = {**cache_kwargs, **kwargs}
                
                # Try to get from cache
                cached_result = await self.get(cache_type, **cache_key_params)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_type, result, **cache_key_params)
                
                return result
            return wrapper
        return decorator
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        redis_info = self.redis_client.info('memory')
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_max': self.max_memory_size,
            'redis_memory_usage': redis_info.get('used_memory_human', 'unknown'),
            'redis_keys': self.redis_client.dbsize()
        }

# Global cache manager
cache_manager = CacheManager()

# Usage examples
@cache_manager.cached('scenario', ttl_hours=24)
async def load_scenario(scenario_id: str):
    # Expensive scenario loading logic
    return {'id': scenario_id, 'data': 'scenario_data'}

@cache_manager.cached('model_response')
async def get_model_response(model: str, prompt: str):
    # Expensive API call
    return {'response': 'model_output'}
```

## Database Optimization

### Query Optimization

```python
# src/agent_skeptic_bench/database_optimization.py
from sqlalchemy import create_engine, text, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

Base = declarative_base()

class OptimizedEvaluationResult(Base):
    """Optimized table schema for evaluation results."""
    __tablename__ = 'evaluation_results'
    
    # Primary key and basic fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Indexed fields for common queries
    model = Column(String(100), index=True, nullable=False)
    category = Column(String(50), index=True, nullable=False)
    scenario_id = Column(String(200), index=True, nullable=False)
    
    # Performance metrics (indexed for aggregations)
    skepticism_score = Column(Float, index=True)
    evaluation_time = Column(Float, index=True)
    api_calls = Column(Integer, index=True)
    
    # Status and metadata
    status = Column(String(20), index=True, default='completed')
    version = Column(String(20), index=True)
    
    # JSON data for flexible storage
    detailed_results = Column(JSONB)
    evaluation_metadata = Column(JSONB)
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_model_category_date', 'model', 'category', 'created_at'),
        Index('idx_performance_lookup', 'model', 'skepticism_score', 'evaluation_time'),
        Index('idx_scenario_model', 'scenario_id', 'model'),
        Index('idx_status_date', 'status', 'created_at'),
        # GIN index for JSONB queries
        Index('idx_detailed_results_gin', 'detailed_results', postgresql_using='gin'),
    )

class DatabaseOptimizer:
    """Database optimization utilities."""
    
    def __init__(self, database_url: str):
        # Connection pool optimized for concurrent workloads
        self.engine = create_engine(
            database_url,
            pool_size=20,  # Base connections
            max_overflow=30,  # Additional connections
            pool_pre_ping=True,  # Verify connections
            pool_recycle=3600,  # Recycle connections hourly
            echo=False  # Set to True for query logging
        )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_optimized_indexes(self):
        """Create performance-optimized indexes."""
        with self.engine.connect() as conn:
            # Additional specialized indexes
            queries = [
                # Partial index for recent evaluations
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_evaluations 
                ON evaluation_results (created_at DESC, model, category) 
                WHERE created_at > NOW() - INTERVAL '30 days'
                """,
                
                # Covering index for leaderboard queries
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leaderboard_covering 
                ON evaluation_results (model, category, skepticism_score DESC) 
                INCLUDE (evaluation_time, api_calls, created_at)
                """,
                
                # Expression index for score ranges
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_score_ranges 
                ON evaluation_results ((CASE 
                    WHEN skepticism_score >= 0.8 THEN 'high'
                    WHEN skepticism_score >= 0.5 THEN 'medium'
                    ELSE 'low'
                END), model)
                """
            ]
            
            for query in queries:
                try:
                    conn.execute(text(query))
                    conn.commit()
                except Exception as e:
                    logging.error(f"Index creation failed: {e}")
    
    def get_optimized_query_plans(self) -> List[Dict[str, Any]]:
        """Analyze query performance and suggest optimizations."""
        with self.engine.connect() as conn:
            # Common query patterns to analyze
            queries = [
                {
                    'name': 'model_performance_summary',
                    'query': """
                    SELECT model, category, 
                           AVG(skepticism_score) as avg_score,
                           COUNT(*) as total_evaluations
                    FROM evaluation_results 
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    GROUP BY model, category
                    ORDER BY avg_score DESC
                    """
                },
                {
                    'name': 'scenario_difficulty_analysis',
                    'query': """
                    SELECT scenario_id,
                           AVG(skepticism_score) as avg_difficulty,
                           AVG(evaluation_time) as avg_time
                    FROM evaluation_results
                    WHERE status = 'completed'
                    GROUP BY scenario_id
                    HAVING COUNT(*) >= 10
                    ORDER BY avg_difficulty ASC
                    """
                },
                {
                    'name': 'recent_evaluations_trend',
                    'query': """
                    SELECT DATE(created_at) as eval_date,
                           model,
                           COUNT(*) as daily_count,
                           AVG(evaluation_time) as avg_time
                    FROM evaluation_results
                    WHERE created_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(created_at), model
                    ORDER BY eval_date DESC, daily_count DESC
                    """
                }
            ]
            
            query_plans = []
            for query_info in queries:
                try:
                    # Get query execution plan
                    plan_result = conn.execute(
                        text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_info['query']}")
                    )
                    plan_data = plan_result.scalar()
                    
                    query_plans.append({
                        'name': query_info['name'],
                        'execution_plan': plan_data,
                        'query': query_info['query']
                    })
                except Exception as e:
                    logging.error(f"Query plan analysis failed for {query_info['name']}: {e}")
            
            return query_plans
    
    def vacuum_and_analyze(self):
        """Perform database maintenance for optimal performance."""
        with self.engine.connect() as conn:
            # Auto-commit mode for VACUUM
            conn.execute(text("COMMIT"))
            
            maintenance_queries = [
                "VACUUM ANALYZE evaluation_results",
                "REINDEX INDEX CONCURRENTLY idx_model_category_date",
                "UPDATE pg_stat_user_tables SET n_tup_ins = 0, n_tup_upd = 0, n_tup_del = 0"
            ]
            
            for query in maintenance_queries:
                try:
                    conn.execute(text(query))
                    logging.info(f"Executed maintenance: {query}")
                except Exception as e:
                    logging.error(f"Maintenance failed: {query} - {e}")
```

## Load Testing

### Performance Benchmarks

```python
# tests/performance/test_load_testing.py
import asyncio
import time
import statistics
from typing import List, Dict, Any
import pytest
from concurrent.futures import ThreadPoolExecutor
from agent_skeptic_bench import AsyncEvaluationEngine, EvaluationTask

class LoadTester:
    """Comprehensive load testing for Agent Skeptic Bench."""
    
    def __init__(self):
        self.engine = AsyncEvaluationEngine(
            max_concurrent_evaluations=50,
            max_api_requests_per_second=100
        )
        self.results: List[Dict[str, Any]] = []
    
    async def run_load_test(self, 
                           num_tasks: int = 1000,
                           ramp_up_time: int = 60,
                           test_duration: int = 300) -> Dict[str, Any]:
        """Run comprehensive load test."""
        
        # Generate test tasks
        tasks = self._generate_test_tasks(num_tasks)
        
        # Ramp up gradually to avoid thundering herd
        task_batches = self._create_ramp_up_batches(tasks, ramp_up_time)
        
        start_time = time.time()
        all_results = []
        
        for batch_time, batch_tasks in task_batches:
            # Wait until it's time for this batch
            current_time = time.time()
            wait_time = batch_time - (current_time - start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Submit batch
            batch_results = await self.engine.evaluate_batch(batch_tasks)
            all_results.extend(batch_results)
            
            # Stop if test duration exceeded
            if time.time() - start_time > test_duration:
                break
        
        # Analyze results
        return self._analyze_load_test_results(all_results, start_time)
    
    def _generate_test_tasks(self, num_tasks: int) -> List[EvaluationTask]:
        """Generate synthetic test tasks."""
        models = ['gpt-4', 'claude-3', 'gemini-pro']
        categories = ['factual_claims', 'flawed_plans', 'persuasion_attacks']
        
        tasks = []
        for i in range(num_tasks):
            task = EvaluationTask(
                scenario_id=f"test_scenario_{i % 100}",
                model=models[i % len(models)],
                category=categories[i % len(categories)],
                priority=1 if i % 10 == 0 else 2  # 10% high priority
            )
            tasks.append(task)
        
        return tasks
    
    def _create_ramp_up_batches(self, tasks: List[EvaluationTask], 
                              ramp_up_time: int) -> List[tuple]:
        """Create batches for gradual ramp-up."""
        batch_size = max(1, len(tasks) // (ramp_up_time // 5))  # Batch every 5 seconds
        batches = []
        
        for i in range(0, len(tasks), batch_size):
            batch_time = (i // batch_size) * 5  # 5-second intervals
            batch_tasks = tasks[i:i + batch_size]
            batches.append((batch_time, batch_tasks))
        
        return batches
    
    def _analyze_load_test_results(self, results: List[Dict[str, Any]], 
                                 start_time: float) -> Dict[str, Any]:
        """Analyze load test performance metrics."""
        successful_results = [r for r in results if r.get('status') == 'success']
        failed_results = [r for r in results if r.get('status') in ['error', 'timeout']]
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Extract response times (simulated)
        response_times = [r.get('response_time', 1.0) for r in successful_results]
        
        total_time = time.time() - start_time
        throughput = len(results) / total_time
        
        analysis = {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'total_duration': total_time,
            'throughput_rps': throughput,
            'response_times': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'mean': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
                'p99': statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0
            },
            'error_breakdown': self._categorize_errors(failed_results)
        }
        
        return analysis
    
    def _categorize_errors(self, failed_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize different types of errors."""
        error_categories = {}
        
        for result in failed_results:
            if result.get('status') == 'timeout':
                error_categories['timeout'] = error_categories.get('timeout', 0) + 1
            elif 'rate_limit' in str(result.get('error', '')).lower():
                error_categories['rate_limit'] = error_categories.get('rate_limit', 0) + 1
            elif 'connection' in str(result.get('error', '')).lower():
                error_categories['connection'] = error_categories.get('connection', 0) + 1
            else:
                error_categories['other'] = error_categories.get('other', 0) + 1
        
        return error_categories

@pytest.mark.performance
@pytest.mark.asyncio
async def test_baseline_performance():
    """Test baseline performance with moderate load."""
    tester = LoadTester()
    results = await tester.run_load_test(
        num_tasks=100,
        ramp_up_time=30,
        test_duration=120
    )
    
    # Performance assertions
    assert results['success_rate'] >= 95.0, f"Success rate too low: {results['success_rate']}%"
    assert results['throughput_rps'] >= 10.0, f"Throughput too low: {results['throughput_rps']} RPS"
    assert results['response_times']['p95'] <= 5.0, f"P95 response time too high: {results['response_times']['p95']}s"

@pytest.mark.performance
@pytest.mark.stress
@pytest.mark.asyncio
async def test_stress_performance():
    """Test performance under stress conditions."""
    tester = LoadTester()
    results = await tester.run_load_test(
        num_tasks=1000,
        ramp_up_time=60,
        test_duration=600
    )
    
    # Stress test assertions (more lenient)
    assert results['success_rate'] >= 90.0, f"Success rate under stress: {results['success_rate']}%"
    assert results['throughput_rps'] >= 5.0, f"Throughput under stress: {results['throughput_rps']} RPS"
    assert results['response_times']['p99'] <= 30.0, f"P99 response time under stress: {results['response_times']['p99']}s"

if __name__ == '__main__':
    # Run performance tests
    asyncio.run(test_baseline_performance())
    asyncio.run(test_stress_performance())
```

## Resource Monitoring

### Performance Dashboards

```python
# src/agent_skeptic_bench/performance_dashboard.py
from flask import Flask, jsonify, render_template_string
import psutil
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

app = Flask(__name__)

class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'per_cpu': psutil.cpu_percent(percpu=True)
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'processes': len(psutil.pids()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        return metrics
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get aggregated metrics for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m['cpu']['percent'] for m in recent_metrics]
        memory_values = [m['memory']['percent'] for m in recent_metrics]
        
        return {
            'time_range_minutes': minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'alerts': self._check_alerts(recent_metrics)
        }
    
    def _check_alerts(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Check for performance alerts."""
        alerts = []
        
        if not metrics:
            return alerts
        
        latest = metrics[-1]
        
        # CPU alerts
        if latest['cpu']['percent'] > 90:
            alerts.append({
                'type': 'critical',
                'metric': 'cpu',
                'message': f"High CPU usage: {latest['cpu']['percent']:.1f}%"
            })
        elif latest['cpu']['percent'] > 80:
            alerts.append({
                'type': 'warning',
                'metric': 'cpu',
                'message': f"Elevated CPU usage: {latest['cpu']['percent']:.1f}%"
            })
        
        # Memory alerts
        if latest['memory']['percent'] > 90:
            alerts.append({
                'type': 'critical',
                'metric': 'memory',
                'message': f"High memory usage: {latest['memory']['percent']:.1f}%"
            })
        elif latest['memory']['percent'] > 80:
            alerts.append({
                'type': 'warning',
                'metric': 'memory',
                'message': f"Elevated memory usage: {latest['memory']['percent']:.1f}%"
            })
        
        # Disk alerts
        if latest['disk']['percent'] > 90:
            alerts.append({
                'type': 'critical',
                'metric': 'disk',
                'message': f"High disk usage: {latest['disk']['percent']:.1f}%"
            })
        
        return alerts

monitor = PerformanceMonitor()

@app.route('/metrics')
def get_metrics():
    """Get current system metrics."""
    return jsonify(monitor.collect_metrics())

@app.route('/metrics/summary')
def get_metrics_summary():
    """Get metrics summary."""
    minutes = request.args.get('minutes', 5, type=int)
    return jsonify(monitor.get_metrics_summary(minutes))

@app.route('/dashboard')
def dashboard():
    """Performance dashboard HTML."""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Skeptic Bench - Performance Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
            .alert-critical { background-color: #ffebee; border-left: 4px solid #f44336; }
            .alert-warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
            .chart-container { width: 45%; display: inline-block; margin: 2%; }
        </style>
    </head>
    <body>
        <h1>Agent Skeptic Bench - Performance Dashboard</h1>
        
        <div id="alerts"></div>
        
        <div class="chart-container">
            <canvas id="cpuChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="memoryChart"></canvas>
        </div>
        
        <div id="metrics"></div>
        
        <script>
            // Initialize charts
            const cpuCtx = document.getElementById('cpuChart').getContext('2d');
            const memoryCtx = document.getElementById('memoryChart').getContext('2d');
            
            const cpuChart = new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            const memoryChart = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Update dashboard every 5 seconds
            function updateDashboard() {
                fetch('/metrics')
                    .then(response => response.json())
                    .then(data => {
                        // Update charts
                        const timestamp = new Date(data.timestamp).toLocaleTimeString();
                        
                        cpuChart.data.labels.push(timestamp);
                        cpuChart.data.datasets[0].data.push(data.cpu.percent);
                        
                        memoryChart.data.labels.push(timestamp);
                        memoryChart.data.datasets[0].data.push(data.memory.percent);
                        
                        // Keep only last 20 data points
                        if (cpuChart.data.labels.length > 20) {
                            cpuChart.data.labels.shift();
                            cpuChart.data.datasets[0].data.shift();
                            memoryChart.data.labels.shift();
                            memoryChart.data.datasets[0].data.shift();
                        }
                        
                        cpuChart.update();
                        memoryChart.update();
                        
                        // Update metrics display
                        document.getElementById('metrics').innerHTML = `
                            <div class="metric-card">
                                <h3>Current Metrics</h3>
                                <p><strong>CPU:</strong> ${data.cpu.percent.toFixed(1)}%</p>
                                <p><strong>Memory:</strong> ${data.memory.percent.toFixed(1)}%</p>
                                <p><strong>Disk:</strong> ${data.disk.percent.toFixed(1)}%</p>
                                <p><strong>Processes:</strong> ${data.processes}</p>
                            </div>
                        `;
                    });
                
                // Update alerts
                fetch('/metrics/summary')
                    .then(response => response.json())
                    .then(data => {
                        const alertsDiv = document.getElementById('alerts');
                        if (data.alerts && data.alerts.length > 0) {
                            alertsDiv.innerHTML = data.alerts.map(alert => 
                                `<div class="metric-card alert-${alert.type}">
                                    <strong>${alert.type.toUpperCase()}:</strong> ${alert.message}
                                </div>`
                            ).join('');
                        } else {
                            alertsDiv.innerHTML = '';
                        }
                    });
            }
            
            // Initial load and set interval
            updateDashboard();
            setInterval(updateDashboard, 5000);
        </script>
    </body>
    </html>
    """
    return dashboard_html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

This comprehensive performance optimization guide provides the foundation for running Agent Skeptic Bench efficiently at scale, with proper monitoring, caching, and optimization strategies.
