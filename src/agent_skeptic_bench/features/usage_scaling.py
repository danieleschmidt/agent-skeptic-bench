"""Auto-scaling and load balancing for usage metrics system."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_depth: int
    active_connections: int
    timestamp: datetime


@dataclass
class ScalingRule:
    """Auto-scaling rule definition."""
    
    metric: str
    threshold: float
    action: str  # "scale_up", "scale_down"
    cooldown_seconds: int
    min_instances: int
    max_instances: int


class AutoScaler:
    """Auto-scaling manager for usage metrics system."""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        """Initialize auto-scaler."""
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Scaling rules
        self.scaling_rules = [
            ScalingRule("cpu_usage", 80.0, "scale_up", 300, min_instances, max_instances),
            ScalingRule("cpu_usage", 30.0, "scale_down", 600, min_instances, max_instances),
            ScalingRule("memory_usage", 85.0, "scale_up", 300, min_instances, max_instances),
            ScalingRule("request_rate", 100.0, "scale_up", 180, min_instances, max_instances),
            ScalingRule("response_time", 5.0, "scale_up", 240, min_instances, max_instances),
            ScalingRule("error_rate", 5.0, "scale_up", 120, min_instances, max_instances)
        ]
        
        self.last_scaling_action = 0
        self.scaling_history: List[Dict[str, Any]] = []
        
        logger.info(f"AutoScaler initialized: {min_instances}-{max_instances} instances")
    
    def evaluate_scaling(self, metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        
        for rule in self.scaling_rules:
            metric_value = getattr(metrics, rule.metric, 0)
            
            # Check if threshold is breached
            should_scale = False
            
            if rule.action == "scale_up" and metric_value > rule.threshold:
                should_scale = True
            elif rule.action == "scale_down" and metric_value < rule.threshold:
                should_scale = True
            
            if not should_scale:
                continue
            
            # Check cooldown period
            if current_time - self.last_scaling_action < rule.cooldown_seconds:
                logger.debug(f"Scaling blocked by cooldown: {rule.metric}")
                continue
            
            # Check instance limits
            if rule.action == "scale_up" and self.current_instances >= rule.max_instances:
                logger.warning(f"Cannot scale up: at maximum instances ({rule.max_instances})")
                continue
            
            if rule.action == "scale_down" and self.current_instances <= rule.min_instances:
                logger.debug(f"Cannot scale down: at minimum instances ({rule.min_instances})")
                continue
            
            # Execute scaling action
            scaling_decision = {
                "action": rule.action,
                "reason": f"{rule.metric} {metric_value:.1f} {'>' if rule.action == 'scale_up' else '<'} {rule.threshold}",
                "current_instances": self.current_instances,
                "target_instances": self.current_instances + (1 if rule.action == "scale_up" else -1),
                "timestamp": datetime.utcnow().isoformat(),
                "triggered_by": rule.metric
            }
            
            # Update instance count
            if rule.action == "scale_up":
                self.current_instances += 1
            else:
                self.current_instances -= 1
            
            self.last_scaling_action = current_time
            self.scaling_history.append(scaling_decision)
            
            # Keep only last 100 scaling events
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]
            
            logger.info(f"Scaling decision: {scaling_decision['action']} to {scaling_decision['target_instances']} instances")
            return scaling_decision
        
        return None
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "last_scaling_action": self.last_scaling_action,
            "recent_history": self.scaling_history[-10:],  # Last 10 scaling events
            "rules_count": len(self.scaling_rules)
        }


class LoadBalancer:
    """Load balancer for distributing usage metrics operations."""
    
    def __init__(self, algorithm: str = "round_robin"):
        """Initialize load balancer."""
        self.algorithm = algorithm
        self.instances: List[Dict[str, Any]] = []
        self.current_index = 0
        self.request_counts = {}
        
        logger.info(f"LoadBalancer initialized: algorithm={algorithm}")
    
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0) -> None:
        """Add instance to load balancer."""
        instance = {
            "id": instance_id,
            "endpoint": endpoint,
            "weight": weight,
            "active": True,
            "health_score": 1.0,
            "request_count": 0,
            "error_count": 0,
            "last_request": 0
        }
        
        self.instances.append(instance)
        self.request_counts[instance_id] = 0
        
        logger.info(f"Added instance {instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str) -> bool:
        """Remove instance from load balancer."""
        for i, instance in enumerate(self.instances):
            if instance["id"] == instance_id:
                del self.instances[i]
                if instance_id in self.request_counts:
                    del self.request_counts[instance_id]
                logger.info(f"Removed instance {instance_id} from load balancer")
                return True
        
        return False
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next instance based on load balancing algorithm."""
        if not self.instances:
            return None
        
        # Filter active and healthy instances
        healthy_instances = [
            inst for inst in self.instances 
            if inst["active"] and inst["health_score"] > 0.5
        ]
        
        if not healthy_instances:
            # Fall back to any active instance
            healthy_instances = [inst for inst in self.instances if inst["active"]]
        
        if not healthy_instances:
            return None
        
        if self.algorithm == "round_robin":
            return self._round_robin_selection(healthy_instances)
        elif self.algorithm == "least_connections":
            return self._least_connections_selection(healthy_instances)
        elif self.algorithm == "weighted":
            return self._weighted_selection(healthy_instances)
        else:
            return healthy_instances[0]  # Default to first instance
    
    def _round_robin_selection(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Round-robin instance selection."""
        if self.current_index >= len(instances):
            self.current_index = 0
        
        instance = instances[self.current_index]
        self.current_index += 1
        
        return instance
    
    def _least_connections_selection(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select instance with least active connections."""
        return min(instances, key=lambda inst: inst["request_count"])
    
    def _weighted_selection(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted selection based on instance weights and health."""
        # Calculate effective weights
        total_weight = sum(inst["weight"] * inst["health_score"] for inst in instances)
        
        if total_weight == 0:
            return instances[0]
        
        # Simple weighted selection (could be more sophisticated)
        import random
        target = random.random() * total_weight
        
        current_weight = 0
        for instance in instances:
            current_weight += instance["weight"] * instance["health_score"]
            if current_weight >= target:
                return instance
        
        return instances[-1]  # Fallback
    
    def record_request(self, instance_id: str, success: bool = True, response_time: float = 0) -> None:
        """Record request metrics for an instance."""
        for instance in self.instances:
            if instance["id"] == instance_id:
                instance["request_count"] += 1
                instance["last_request"] = time.time()
                
                if not success:
                    instance["error_count"] += 1
                
                # Update health score based on recent performance
                error_rate = instance["error_count"] / max(instance["request_count"], 1)
                instance["health_score"] = max(0.1, 1.0 - error_rate)
                
                self.request_counts[instance_id] += 1
                break
    
    def get_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        if not self.instances:
            return {"error": "No instances available"}
        
        total_requests = sum(inst["request_count"] for inst in self.instances)
        total_errors = sum(inst["error_count"] for inst in self.instances)
        
        return {
            "algorithm": self.algorithm,
            "total_instances": len(self.instances),
            "healthy_instances": sum(1 for inst in self.instances if inst["health_score"] > 0.5),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / max(total_requests, 1),
            "instances": [
                {
                    "id": inst["id"],
                    "active": inst["active"],
                    "health_score": inst["health_score"],
                    "request_count": inst["request_count"],
                    "error_rate": inst["error_count"] / max(inst["request_count"], 1)
                }
                for inst in self.instances
            ]
        }


class ResourceManager:
    """Manages system resources for optimal performance."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.resource_limits = {
            "max_memory_mb": 1024,
            "max_cpu_percent": 80,
            "max_disk_mb": 2048,
            "max_concurrent_operations": 50
        }
        
        self.current_usage = {
            "memory_mb": 0,
            "cpu_percent": 0,
            "disk_mb": 0,
            "concurrent_operations": 0
        }
        
        self.last_check = time.time()
        
        logger.info("ResourceManager initialized")
    
    def check_resource_availability(self, operation_type: str) -> Dict[str, bool]:
        """Check if resources are available for operation."""
        self._update_current_usage()
        
        availability = {
            "memory_available": self.current_usage["memory_mb"] < self.resource_limits["max_memory_mb"] * 0.9,
            "cpu_available": self.current_usage["cpu_percent"] < self.resource_limits["max_cpu_percent"],
            "disk_available": self.current_usage["disk_mb"] < self.resource_limits["max_disk_mb"] * 0.9,
            "concurrency_available": self.current_usage["concurrent_operations"] < self.resource_limits["max_concurrent_operations"]
        }
        
        availability["overall_available"] = all(availability.values())
        
        return availability
    
    def _update_current_usage(self) -> None:
        """Update current resource usage metrics."""
        current_time = time.time()
        
        # Only update every 10 seconds to avoid overhead
        if current_time - self.last_check < 10:
            return
        
        try:
            # Simulate resource usage measurement
            # In real implementation, this would query actual system metrics
            import random
            
            self.current_usage.update({
                "memory_mb": random.randint(100, 800),
                "cpu_percent": random.randint(10, 70),
                "disk_mb": random.randint(50, 1500),
                "concurrent_operations": len(self.current_usage)  # Simplified
            })
            
            self.last_check = current_time
            
        except Exception as e:
            logger.error(f"Failed to update resource usage: {e}")
    
    def reserve_resources(self, operation_type: str, estimated_memory: int = 50, 
                         estimated_cpu: int = 10) -> str:
        """Reserve resources for an operation."""
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        
        # Check availability
        availability = self.check_resource_availability(operation_type)
        
        if not availability["overall_available"]:
            logger.warning(f"Insufficient resources for {operation_type}")
            return ""
        
        # Reserve resources (simplified)
        self.current_usage["concurrent_operations"] += 1
        
        logger.debug(f"Reserved resources for {operation_id}")
        return operation_id
    
    def release_resources(self, operation_id: str) -> None:
        """Release reserved resources."""
        if self.current_usage["concurrent_operations"] > 0:
            self.current_usage["concurrent_operations"] -= 1
        
        logger.debug(f"Released resources for {operation_id}")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        self._update_current_usage()
        
        return {
            "current_usage": self.current_usage.copy(),
            "resource_limits": self.resource_limits.copy(),
            "utilization": {
                name: (self.current_usage[name] / self.resource_limits[f"max_{name}"]) * 100
                for name in ["memory_mb", "cpu_percent", "disk_mb"]
            },
            "availability": self.check_resource_availability("general")
        }


class QueueManager:
    """Manages request queues for high-load scenarios."""
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize queue manager."""
        self.max_queue_size = max_queue_size
        self.priority_queue: List[Tuple[int, str, Dict[str, Any]]] = []  # (priority, id, data)
        self.processing_queue: List[Tuple[str, Dict[str, Any]]] = []
        self.completed_requests = {}
        
        logger.info(f"QueueManager initialized: max_size={max_queue_size}")
    
    def enqueue_request(self, request_id: str, request_data: Dict[str, Any], 
                       priority: int = 5) -> bool:
        """Enqueue a request for processing."""
        if len(self.priority_queue) >= self.max_queue_size:
            logger.warning("Queue at capacity, rejecting request")
            return False
        
        # Add to priority queue
        self.priority_queue.append((priority, request_id, request_data))
        
        # Sort by priority (lower numbers = higher priority)
        self.priority_queue.sort(key=lambda x: x[0])
        
        logger.debug(f"Enqueued request {request_id} with priority {priority}")
        return True
    
    def dequeue_request(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Dequeue highest priority request."""
        if not self.priority_queue:
            return None
        
        priority, request_id, request_data = self.priority_queue.pop(0)
        
        # Move to processing queue
        self.processing_queue.append((request_id, request_data))
        
        logger.debug(f"Dequeued request {request_id} (priority {priority})")
        return request_id, request_data
    
    def complete_request(self, request_id: str, result: Any) -> None:
        """Mark request as completed."""
        # Remove from processing queue
        self.processing_queue = [
            (rid, data) for rid, data in self.processing_queue 
            if rid != request_id
        ]
        
        # Store result temporarily
        self.completed_requests[request_id] = {
            "result": result,
            "completed_at": time.time()
        }
        
        # Cleanup old completed requests (keep last 1000)
        if len(self.completed_requests) > 1000:
            oldest_requests = sorted(
                self.completed_requests.items(),
                key=lambda x: x[1]["completed_at"]
            )[:100]  # Remove oldest 100
            
            for old_request_id, _ in oldest_requests:
                del self.completed_requests[old_request_id]
        
        logger.debug(f"Completed request {request_id}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending_requests": len(self.priority_queue),
            "processing_requests": len(self.processing_queue),
            "completed_requests": len(self.completed_requests),
            "max_queue_size": self.max_queue_size,
            "queue_utilization": len(self.priority_queue) / self.max_queue_size,
            "average_priority": sum(p for p, _, _ in self.priority_queue) / max(len(self.priority_queue), 1)
        }


class PerformanceMonitor:
    """Monitors system performance for scaling decisions."""
    
    def __init__(self, monitoring_interval: int = 30):
        """Initialize performance monitor."""
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[ScalingMetrics] = []
        self.last_monitoring = time.time()
        
        # Performance thresholds for alerts
        self.thresholds = {
            "cpu_critical": 90.0,
            "memory_critical": 95.0,
            "response_time_critical": 10.0,
            "error_rate_critical": 10.0
        }
        
        logger.info(f"PerformanceMonitor initialized: interval={monitoring_interval}s")
    
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current performance metrics."""
        import random
        
        # Simulate metrics collection
        # In real implementation, this would gather actual system metrics
        metrics = ScalingMetrics(
            cpu_usage=random.uniform(20, 85),
            memory_usage=random.uniform(30, 90),
            request_rate=random.uniform(10, 150),
            response_time=random.uniform(0.5, 8.0),
            error_rate=random.uniform(0, 8),
            queue_depth=random.randint(0, 50),
            active_connections=random.randint(5, 200),
            timestamp=datetime.utcnow()
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        self.last_monitoring = time.time()
        
        return metrics
    
    def check_alerts(self, metrics: ScalingMetrics) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.thresholds["cpu_critical"]:
            alerts.append({
                "type": "cpu_critical",
                "message": f"CPU usage critical: {metrics.cpu_usage:.1f}%",
                "severity": "critical",
                "timestamp": metrics.timestamp.isoformat()
            })
        
        if metrics.memory_usage > self.thresholds["memory_critical"]:
            alerts.append({
                "type": "memory_critical",
                "message": f"Memory usage critical: {metrics.memory_usage:.1f}%",
                "severity": "critical",
                "timestamp": metrics.timestamp.isoformat()
            })
        
        if metrics.response_time > self.thresholds["response_time_critical"]:
            alerts.append({
                "type": "response_time_critical",
                "message": f"Response time critical: {metrics.response_time:.1f}s",
                "severity": "warning",
                "timestamp": metrics.timestamp.isoformat()
            })
        
        if metrics.error_rate > self.thresholds["error_rate_critical"]:
            alerts.append({
                "type": "error_rate_critical",
                "message": f"Error rate critical: {metrics.error_rate:.1f}%",
                "severity": "critical",
                "timestamp": metrics.timestamp.isoformat()
            })
        
        return alerts
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics data available"}
        
        # Calculate trends
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        response_times = [m.response_time for m in recent_metrics]
        
        return {
            "time_window_hours": hours,
            "data_points": len(recent_metrics),
            "cpu_trend": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory_trend": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": memory_values[-1] if memory_values else 0
            },
            "response_time_trend": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "current": response_times[-1] if response_times else 0
            }
        }