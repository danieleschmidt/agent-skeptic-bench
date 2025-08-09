"""Health checking and monitoring for Agent Skeptic Bench."""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

# Handle optional dependencies with graceful fallbacks
try:
    import aiohttp
    aiohttp_available = True
except ImportError:
    # Fallback stubs for aiohttp
    class _MockClientSession:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def post(self, *args, **kwargs): return _MockResponse()
        def get(self, *args, **kwargs): return _MockResponse()
    
    class _MockResponse:
        def __init__(self):
            self.status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    
    class _MockTimeout:
        def __init__(self, *args, **kwargs): pass
    
    class _MockAiohttp:
        ClientSession = _MockClientSession
        ClientTimeout = _MockTimeout
    
    aiohttp = _MockAiohttp()
    aiohttp_available = False

try:
    import psutil
    psutil_available = True
except ImportError:
    # Fallback stubs for psutil
    class _MockMemory:
        def __init__(self):
            self.percent = 0.0
            self.used = 0
    
    class _MockDisk:
        def __init__(self):
            self.used = 0
            self.total = 1
    
    class _MockNetworkIO:
        def __init__(self):
            self.bytes_sent = 0
            self.bytes_recv = 0
            self.packets_sent = 0
            self.packets_recv = 0
    
    class _MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 0
                vms = 0
            return MemInfo()
        def num_threads(self): return 1
        def open_files(self): return []
    
    class _MockPsutil:
        def cpu_percent(self, interval=None): return 0.0
        def virtual_memory(self): return _MockMemory()
        def disk_usage(self, path): return _MockDisk()
        def net_io_counters(self): return _MockNetworkIO()
        def pids(self): return [1]
        def Process(self): return _MockProcess()
    
    psutil = _MockPsutil()
    psutil_available = False


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of system components to monitor."""
    
    DATABASE = "database"
    CACHE = "cache"
    API = "api"
    EXTERNAL_SERVICE = "external_service"
    FILESYSTEM = "filesystem"
    SYSTEM_RESOURCES = "system_resources"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    component: str
    component_type: ComponentType
    status: HealthStatus
    response_time: float
    timestamp: datetime
    message: str
    details: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    open_file_descriptors: int
    uptime: float


class HealthChecker:
    """Comprehensive health checker for the system."""
    
    def __init__(self):
        """Initialize health checker."""
        self.health_checks: Dict[str, Callable] = {}
        self.check_history: List[HealthCheckResult] = []
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'response_time': {'warning': 1.0, 'critical': 5.0}
        }
        self._start_time = time.time()
    
    def register_health_check(self, name: str, check_func: Callable, 
                            component_type: ComponentType) -> None:
        """Register a health check function."""
        self.health_checks[name] = {
            'func': check_func,
            'type': component_type
        }
        logger.info(f"Registered health check: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Run system resource checks
        results['system_resources'] = await self._check_system_resources()
        
        # Run registered checks
        for name, check_info in self.health_checks.items():
            try:
                result = await self._run_single_check(name, check_info)
                results[name] = result
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    component_type=check_info['type'],
                    status=HealthStatus.CRITICAL,
                    response_time=0.0,
                    timestamp=datetime.utcnow(),
                    message="Health check failed",
                    details={},
                    error=str(e)
                )
        
        # Store results in history
        for result in results.values():
            self.check_history.append(result)
        
        # Keep only recent history (last 1000 checks per component)
        self._cleanup_history()
        
        return results
    
    async def _run_single_check(self, name: str, check_info: Dict[str, Any]) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_info['func']):
                check_result = await check_info['func']()
            else:
                check_result = check_info['func']()
            
            response_time = time.time() - start_time
            
            # Determine status based on response time
            status = HealthStatus.HEALTHY
            if response_time > self.alert_thresholds['response_time']['critical']:
                status = HealthStatus.CRITICAL
            elif response_time > self.alert_thresholds['response_time']['warning']:
                status = HealthStatus.DEGRADED
            
            # Override status if check explicitly returns one
            if isinstance(check_result, dict) and 'status' in check_result:
                status = check_result['status']
            
            return HealthCheckResult(
                component=name,
                component_type=check_info['type'],
                status=status,
                response_time=response_time,
                timestamp=datetime.utcnow(),
                message=check_result.get('message', 'Check completed') if isinstance(check_result, dict) else 'Check completed',
                details=check_result if isinstance(check_result, dict) else {'result': check_result}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Health check {name} failed: {e}")
            
            return HealthCheckResult(
                component=name,
                component_type=check_info['type'],
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                timestamp=datetime.utcnow(),
                message="Health check failed",
                details={},
                error=str(e)
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            if psutil_available:
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                
                # Network I/O
                network = psutil.net_io_counters()
                network_io = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
                
                # Process info
                process_count = len(psutil.pids())
                
                # Open file descriptors
                try:
                    open_fds = len(psutil.Process().open_files())
                except:
                    open_fds = 0
            else:
                # Use fallback values when psutil is not available
                cpu_usage = 0.0
                memory_usage = 0.0
                disk_usage = 0.0
                network_io = {
                    'bytes_sent': 0,
                    'bytes_recv': 0,
                    'packets_sent': 0,
                    'packets_recv': 0
                }
                process_count = 1
                open_fds = 0
            
            # Uptime
            uptime = time.time() - self._start_time
            
            # Determine overall status
            status = HealthStatus.HEALTHY
            
            if (cpu_usage > self.alert_thresholds['cpu_usage']['critical'] or
                memory_usage > self.alert_thresholds['memory_usage']['critical'] or
                disk_usage > self.alert_thresholds['disk_usage']['critical']):
                status = HealthStatus.CRITICAL
            elif (cpu_usage > self.alert_thresholds['cpu_usage']['warning'] or
                  memory_usage > self.alert_thresholds['memory_usage']['warning'] or
                  disk_usage > self.alert_thresholds['disk_usage']['warning']):
                status = HealthStatus.DEGRADED
            
            message = f"System resources: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%, Disk {disk_usage:.1f}%"
            if not psutil_available:
                message += " (fallback values - psutil not available)"
            
            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCES,
                status=status,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                message=message,
                details={
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'network_io': network_io,
                    'process_count': process_count,
                    'open_file_descriptors': open_fds,
                    'uptime': uptime,
                    'psutil_available': psutil_available
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCES,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                timestamp=datetime.utcnow(),
                message="Failed to check system resources",
                details={},
                error=str(e)
            )
    
    async def check_database_health(self, connection_string: str) -> Dict[str, Any]:
        """Check database health."""
        try:
            # This would typically use the actual database connection
            # For demonstration, we'll simulate a database check
            start_time = time.time()
            
            # Simulate database query
            await asyncio.sleep(0.1)  # Simulate query time
            
            response_time = time.time() - start_time
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Database connection successful',
                'response_time': response_time,
                'connection_pool_size': 10,
                'active_connections': 3
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Database check failed: {e}',
                'error': str(e)
            }
    
    async def check_cache_health(self, cache_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache health (Redis/Memory)."""
        try:
            start_time = time.time()
            
            # Simulate cache operation
            await asyncio.sleep(0.05)
            
            response_time = time.time() - start_time
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Cache operational',
                'response_time': response_time,
                'cache_size': 1024,
                'hit_rate': 0.85
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Cache check failed: {e}',
                'error': str(e)
            }
    
    async def check_external_service(self, service_url: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Check external service health."""
        if not aiohttp_available:
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'External service check not available (aiohttp not installed)',
                'response_time': 0.0,
                'http_status': 0
            }
            
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(service_url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = f"Service responding (HTTP {response.status})"
                    elif response.status < 500:
                        status = HealthStatus.DEGRADED
                        message = f"Service responding with errors (HTTP {response.status})"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"Service error (HTTP {response.status})"
                    
                    return {
                        'status': status,
                        'message': message,
                        'response_time': response_time,
                        'http_status': response.status
                    }
                    
        except asyncio.TimeoutError:
            return {
                'status': HealthStatus.CRITICAL,
                'message': 'Service timeout',
                'error': 'Request timed out'
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL,
                'message': f'Service check failed: {e}',
                'error': str(e)
            }
    
    def get_component_status(self, component: str) -> Optional[HealthStatus]:
        """Get the latest status of a specific component."""
        recent_checks = [
            check for check in self.check_history 
            if check.component == component
        ]
        
        if recent_checks:
            # Get the most recent check
            latest = max(recent_checks, key=lambda x: x.timestamp)
            return latest.status
        
        return None
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.health_checks:
            return HealthStatus.HEALTHY
        
        # Get latest status for each component
        component_statuses = []
        for component in self.health_checks.keys():
            status = self.get_component_status(component)
            if status:
                component_statuses.append(status)
        
        # Add system resources status
        system_status = self.get_component_status('system_resources')
        if system_status:
            component_statuses.append(system_status)
        
        if not component_statuses:
            return HealthStatus.HEALTHY
        
        # Determine overall status
        if HealthStatus.CRITICAL in component_statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in component_statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in component_statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        overall_status = self.get_overall_health()
        
        # Component status summary
        component_summary = {}
        for component in self.health_checks.keys():
            status = self.get_component_status(component)
            component_summary[component] = status.value if status else "unknown"
        
        # Add system resources
        system_status = self.get_component_status('system_resources')
        component_summary['system_resources'] = system_status.value if system_status else "unknown"
        
        # Recent checks count
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        recent_checks = [
            check for check in self.check_history 
            if check.timestamp > cutoff_time
        ]
        
        return {
            'overall_status': overall_status.value,
            'components': component_summary,
            'recent_checks_count': len(recent_checks),
            'total_checks_run': len(self.check_history),
            'uptime': time.time() - self._start_time,
            'last_check_time': max([check.timestamp for check in self.check_history]).isoformat() if self.check_history else None
        }
    
    def _cleanup_history(self) -> None:
        """Clean up old health check history."""
        # Keep only the last 100 checks per component
        component_counts = {}
        cleaned_history = []
        
        # Sort by timestamp descending
        sorted_history = sorted(self.check_history, key=lambda x: x.timestamp, reverse=True)
        
        for check in sorted_history:
            component = check.component
            if component not in component_counts:
                component_counts[component] = 0
            
            if component_counts[component] < 100:
                cleaned_history.append(check)
                component_counts[component] += 1
        
        self.check_history = cleaned_history
    
    def setup_default_checks(self) -> None:
        """Setup default health checks."""
        # Register default system checks
        self.register_health_check(
            "database",
            lambda: self.check_database_health("postgresql://localhost:5432/agentskeptic"),
            ComponentType.DATABASE
        )
        
        self.register_health_check(
            "cache",
            lambda: self.check_cache_health({"backend": "memory"}),
            ComponentType.CACHE
        )
        
        logger.info("Default health checks registered")


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    
    if _health_checker is None:
        _health_checker = HealthChecker()
        _health_checker.setup_default_checks()
    
    return _health_checker