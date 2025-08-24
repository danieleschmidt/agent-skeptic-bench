"""Optimized usage metrics manager with auto-scaling and high performance."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .usage_cache import UsageMetricsCache, BatchProcessor, AsyncQueryProcessor, PerformanceOptimizer
from .usage_scaling import AutoScaler, LoadBalancer, PerformanceMonitor, ResourceManager, QueueManager, ScalingMetrics
from .analytics import UsageTracker, UsageMetrics

logger = logging.getLogger(__name__)


class OptimizedUsageManager:
    """High-performance usage metrics manager with auto-scaling."""
    
    def __init__(self, 
                 storage_path: str = "data/usage_metrics",
                 enable_caching: bool = True,
                 enable_auto_scaling: bool = True,
                 max_instances: int = 10):
        """Initialize optimized usage manager."""
        
        # Core components
        self.usage_tracker = UsageTracker(storage_path)
        
        # Performance optimization
        self.cache = UsageMetricsCache() if enable_caching else None
        self.batch_processor = BatchProcessor(batch_size=50, flush_interval=15)
        self.query_processor = AsyncQueryProcessor(max_concurrent_queries=20)
        self.performance_optimizer = PerformanceOptimizer()
        
        # Scaling and load management
        if enable_auto_scaling:
            self.auto_scaler = AutoScaler(min_instances=1, max_instances=max_instances)
            self.load_balancer = LoadBalancer(algorithm="least_connections")
            self.performance_monitor = PerformanceMonitor(monitoring_interval=20)
            self.resource_manager = ResourceManager()
            self.queue_manager = QueueManager(max_queue_size=500)
            
            # Initialize with at least one instance
            self.load_balancer.add_instance("primary", "localhost:8000", weight=1.0)
        else:
            self.auto_scaler = None
            self.load_balancer = None
            self.performance_monitor = None
            self.resource_manager = None
            self.queue_manager = None
        
        # Background task management
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_requested = False
        
        logger.info(f"OptimizedUsageManager initialized: caching={enable_caching}, scaling={enable_auto_scaling}")
    
    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        if self.auto_scaler:
            # Performance monitoring task
            self.background_tasks.append(
                asyncio.create_task(self._performance_monitoring_loop())
            )
            
            # Auto-scaling task
            self.background_tasks.append(
                asyncio.create_task(self._auto_scaling_loop())
            )
            
            # Resource cleanup task
            self.background_tasks.append(
                asyncio.create_task(self._cleanup_loop())
            )
        
        # Cache maintenance task
        if self.cache:
            self.background_tasks.append(
                asyncio.create_task(self._cache_maintenance_loop())
            )
        
        # Batch flush task
        self.background_tasks.append(
            asyncio.create_task(self._batch_flush_loop())
        )
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the manager."""
        self.shutdown_requested = True
        
        logger.info("Shutting down OptimizedUsageManager...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Flush any pending batches
        if self.batch_processor:
            self.batch_processor.force_flush()
        
        logger.info("OptimizedUsageManager shutdown complete")
    
    async def create_optimized_session(self, session_id: str, user_id: Optional[str] = None,
                                     agent_provider: Optional[str] = None, 
                                     model: Optional[str] = None) -> Dict[str, Any]:
        """Create session with optimization and resource management."""
        start_time = time.time()
        
        try:
            # Check resource availability
            if self.resource_manager:
                availability = self.resource_manager.check_resource_availability("session_creation")
                if not availability["overall_available"]:
                    return {
                        "success": False,
                        "error": "Insufficient resources",
                        "resource_status": availability
                    }
                
                # Reserve resources
                operation_id = self.resource_manager.reserve_resources("session_creation")
            
            # Create session through usage tracker
            self.usage_tracker.start_session(session_id, user_id, agent_provider, model)
            
            # Cache session info if caching enabled
            if self.cache:
                session_data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "agent_provider": agent_provider,
                    "model": model,
                    "created_at": datetime.utcnow().isoformat()
                }
                self.cache.set_session_metrics(session_id, session_data)
            
            duration = time.time() - start_time
            self.performance_optimizer.record_operation("session_creation", duration)
            
            logger.info(f"Optimized session created: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "creation_time": duration,
                "cached": self.cache is not None
            }
        
        except Exception as e:
            logger.error(f"Failed to create optimized session: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Release resources
            if self.resource_manager and 'operation_id' in locals():
                self.resource_manager.release_resources(operation_id)
    
    async def record_evaluation_optimized(self, session_id: str, scenario_id: str, 
                                        category: str, duration: float, score: float,
                                        tokens_used: int = 0) -> Dict[str, Any]:
        """Record evaluation with optimization and batching."""
        start_time = time.time()
        
        try:
            # Check if this is a high-priority operation
            priority = 3 if duration > 5.0 else 5  # Higher priority for slow evaluations
            
            # Queue the evaluation if queue manager available
            if self.queue_manager:
                request_data = {
                    "type": "evaluation",
                    "session_id": session_id,
                    "scenario_id": scenario_id,
                    "category": category,
                    "duration": duration,
                    "score": score,
                    "tokens_used": tokens_used
                }
                
                request_id = f"eval_{session_id}_{int(time.time() * 1000)}"
                
                if not self.queue_manager.enqueue_request(request_id, request_data, priority):
                    return {"success": False, "error": "Queue full"}
                
                # Process from queue
                queued_request = self.queue_manager.dequeue_request()
                if queued_request:
                    request_id, request_data = queued_request
                    
                    # Execute the evaluation recording
                    self.usage_tracker.record_evaluation(
                        request_data["session_id"],
                        request_data["scenario_id"], 
                        request_data["category"],
                        request_data["duration"],
                        request_data["score"],
                        request_data["tokens_used"]
                    )
                    
                    self.queue_manager.complete_request(request_id, {"success": True})
            
            else:
                # Direct recording without queue
                self.usage_tracker.record_evaluation(session_id, scenario_id, category, duration, score, tokens_used)
            
            # Record performance
            operation_duration = time.time() - start_time
            self.performance_optimizer.record_operation("record_evaluation", operation_duration, 1)
            
            return {
                "success": True,
                "processing_time": operation_duration,
                "queued": self.queue_manager is not None
            }
        
        except Exception as e:
            logger.error(f"Failed to record optimized evaluation: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_usage_summary_optimized(self, days: int = 7, 
                                        use_cache: bool = True) -> Dict[str, Any]:
        """Get usage summary with caching and optimization."""
        start_time = time.time()
        
        cache_key = f"summary_{days}d_{int(datetime.utcnow().timestamp() // 300)}"  # 5-minute cache buckets
        
        # Try cache first
        if use_cache and self.cache:
            cached_result = self.cache.get_usage_summary(cache_key)
            if cached_result:
                logger.debug(f"Returning cached usage summary for {days} days")
                return {
                    "success": True,
                    "data": cached_result,
                    "from_cache": True,
                    "processing_time": time.time() - start_time
                }
        
        # Process query
        try:
            query_params = {"type": "summary", "days": days}
            result = await self.query_processor.process_usage_query(query_params)
            
            # Cache the result
            if self.cache:
                self.cache.set_usage_summary(cache_key, result, ttl=300)
            
            # Record performance
            operation_duration = time.time() - start_time
            self.performance_optimizer.record_operation("get_usage_summary", operation_duration)
            
            return {
                "success": True,
                "data": result,
                "from_cache": False,
                "processing_time": operation_duration
            }
        
        except Exception as e:
            logger.error(f"Failed to get optimized usage summary: {e}")
            return {"success": False, "error": str(e)}
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while not self.shutdown_requested:
            try:
                # Collect current metrics
                metrics = self.performance_monitor.collect_metrics()
                
                # Check for alerts
                alerts = self.performance_monitor.check_alerts(metrics)
                
                if alerts:
                    for alert in alerts:
                        logger.warning(f"Performance alert: {alert['message']}")
                
                # Record load balancer performance
                if self.load_balancer:
                    instance = self.load_balancer.get_next_instance()
                    if instance:
                        # Simulate response time measurement
                        response_time = metrics.response_time
                        success = response_time < 5.0
                        
                        self.load_balancer.record_request(
                            instance["id"], 
                            success=success, 
                            response_time=response_time
                        )
                
                await asyncio.sleep(self.performance_monitor.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _auto_scaling_loop(self) -> None:
        """Background task for auto-scaling decisions."""
        while not self.shutdown_requested:
            try:
                if not self.performance_monitor or not self.auto_scaler:
                    await asyncio.sleep(60)
                    continue
                
                # Get latest metrics
                if self.performance_monitor.metrics_history:
                    latest_metrics = self.performance_monitor.metrics_history[-1]
                    
                    # Evaluate scaling decision
                    scaling_decision = self.auto_scaler.evaluate_scaling(latest_metrics)
                    
                    if scaling_decision:
                        logger.info(f"Auto-scaling triggered: {scaling_decision}")
                        
                        # In a real implementation, this would trigger actual scaling
                        # For now, we just log the decision
                        if scaling_decision["action"] == "scale_up":
                            new_instance_id = f"auto_instance_{int(time.time())}"
                            self.load_balancer.add_instance(
                                new_instance_id, 
                                f"localhost:800{scaling_decision['target_instances']}"
                            )
                        elif scaling_decision["action"] == "scale_down" and len(self.load_balancer.instances) > 1:
                            # Remove least used instance
                            least_used = min(
                                self.load_balancer.instances,
                                key=lambda inst: inst["request_count"]
                            )
                            self.load_balancer.remove_instance(least_used["id"])
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Background task for resource cleanup."""
        while not self.shutdown_requested:
            try:
                cleanup_stats = {}
                
                # Cache cleanup
                if self.cache:
                    cache_stats = self.cache.cleanup_expired()
                    cleanup_stats["cache"] = cache_stats
                
                # Connection pool cleanup
                if self.load_balancer:
                    # Simulate connection cleanup
                    cleanup_stats["connections"] = 0
                
                # Resource manager cleanup
                if self.resource_manager:
                    resource_stats = self.resource_manager.get_resource_stats()
                    cleanup_stats["resources"] = resource_stats
                
                # Log cleanup if significant
                total_cleaned = sum(v for v in cleanup_stats.values() if isinstance(v, int))
                if total_cleaned > 0:
                    logger.info(f"Cleanup completed: {cleanup_stats}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)
    
    async def _cache_maintenance_loop(self) -> None:
        """Background task for cache maintenance."""
        while not self.shutdown_requested:
            try:
                if self.cache:
                    # Get cache stats
                    stats = self.cache.get_cache_stats()
                    
                    # Log cache performance
                    if stats["total_hits"] + stats["total_misses"] > 0:
                        hit_rate = stats["hit_rate"]
                        logger.debug(f"Cache hit rate: {hit_rate:.1%}")
                        
                        # Alert on low hit rate
                        if hit_rate < 0.5 and stats["total_hits"] + stats["total_misses"] > 100:
                            logger.warning(f"Low cache hit rate: {hit_rate:.1%}")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                await asyncio.sleep(120)
    
    async def _batch_flush_loop(self) -> None:
        """Background task for batch processing."""
        while not self.shutdown_requested:
            try:
                # Force flush if interval exceeded
                buffer_stats = self.batch_processor.get_buffer_stats()
                
                if buffer_stats["time_since_last_flush"] > buffer_stats["flush_interval"]:
                    flushed_count = self.batch_processor.force_flush()
                    if flushed_count > 0:
                        logger.debug(f"Background flush: {flushed_count} metrics")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Batch flush error: {e}")
                await asyncio.sleep(30)
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - (getattr(self, 'start_time', time.time())),
            "components": {
                "usage_tracker": "active",
                "cache": "active" if self.cache else "disabled",
                "auto_scaling": "active" if self.auto_scaler else "disabled"
            }
        }
        
        # Add cache statistics
        if self.cache:
            status["cache_stats"] = self.cache.get_cache_stats()
        
        # Add scaling status
        if self.auto_scaler:
            status["scaling_status"] = self.auto_scaler.get_scaling_status()
        
        # Add load balancer stats
        if self.load_balancer:
            status["load_balancer_stats"] = self.load_balancer.get_balancer_stats()
        
        # Add queue stats
        if self.queue_manager:
            status["queue_stats"] = self.queue_manager.get_queue_stats()
        
        # Add performance stats
        status["performance_stats"] = self.performance_optimizer.get_performance_summary()
        
        # Add resource stats
        if self.resource_manager:
            status["resource_stats"] = self.resource_manager.get_resource_stats()
        
        return status
    
    async def export_with_optimization(self, export_params: Dict[str, Any]) -> Dict[str, Any]:
        """Export usage data with performance optimization."""
        start_time = time.time()
        
        try:
            # Check cache for recent exports
            cache_key = f"export_{hash(str(sorted(export_params.items())))}"
            
            if self.cache:
                cached_result = self.cache.get_query_result(cache_key)
                if cached_result:
                    return {
                        "success": True,
                        "data": cached_result,
                        "from_cache": True,
                        "processing_time": time.time() - start_time
                    }
            
            # Queue export request if queue manager available
            if self.queue_manager:
                request_id = f"export_{int(time.time() * 1000)}"
                priority = 2  # High priority for exports
                
                if not self.queue_manager.enqueue_request(request_id, export_params, priority):
                    return {"success": False, "error": "Export queue full"}
            
            # Process export (simplified for demo)
            export_result = {
                "format": export_params.get("format", "json"),
                "days": export_params.get("days", 7),
                "records_count": 150,  # Simulated
                "file_path": f"exports/optimized_export_{int(time.time())}.json",
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Cache result
            if self.cache:
                self.cache.set_query_result(cache_key, export_result)
            
            # Record performance
            operation_duration = time.time() - start_time
            self.performance_optimizer.record_operation("export", operation_duration, 150)
            
            return {
                "success": True,
                "data": export_result,
                "from_cache": False,
                "processing_time": operation_duration
            }
        
        except Exception as e:
            logger.error(f"Optimized export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get system optimization recommendations."""
        recommendations = []
        
        # Performance recommendations
        perf_recs = self.performance_optimizer.get_optimization_recommendations()
        recommendations.extend(perf_recs)
        
        # Cache recommendations
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            
            if cache_stats["hit_rate"] < 0.6:
                recommendations.append({
                    "type": "cache",
                    "issue": "low_hit_rate",
                    "current_rate": cache_stats["hit_rate"],
                    "recommendation": "Consider increasing cache TTL or size"
                })
        
        # Scaling recommendations
        if self.auto_scaler:
            scaling_status = self.auto_scaler.get_scaling_status()
            
            if scaling_status["current_instances"] == scaling_status["max_instances"]:
                recommendations.append({
                    "type": "scaling",
                    "issue": "max_instances_reached",
                    "recommendation": "Consider increasing max_instances limit"
                })
        
        return recommendations