# Performance testing configuration for Agent Skeptic Bench
import pytest
import time
from typing import Generator, Any
from unittest.mock import Mock

@pytest.fixture
def benchmark_config():
    """Standard benchmark configuration for performance tests."""
    return {
        "timeout": 30,
        "max_iterations": 1000,
        "target_percentile": 0.95,
        "acceptable_latency": 0.5,  # seconds
        "memory_limit": 1024 * 1024 * 100,  # 100MB
    }

@pytest.fixture
def mock_model():
    """Mock AI model for performance testing."""
    mock = Mock()
    
    def mock_generate(prompt: str, **kwargs) -> str:
        # Simulate realistic response time
        time.sleep(0.1)
        return f"Mock response to: {prompt[:50]}..."
    
    mock.generate = mock_generate
    return mock

@pytest.fixture
def large_scenario_dataset():
    """Generate large dataset for load testing."""
    scenarios = []
    for i in range(1000):
        scenarios.append({
            "id": f"scenario_{i}",
            "category": "factual_claims" if i % 3 == 0 else "flawed_plans",
            "adversary_claim": f"This is test claim number {i} with enough content to simulate realistic scenarios.",
            "expected_skepticism": 0.7 + (i % 4) * 0.075,  # Vary between 0.7-0.925
            "complexity": "high" if i % 10 == 0 else "medium"
        })
    return scenarios

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    import threading
    
    class MemoryMonitor:
        def __init__(self):
            self.max_memory = 0
            self.monitoring = False
            self.process = psutil.Process()
        
        def start(self):
            self.monitoring = True
            self.max_memory = self.process.memory_info().rss
            threading.Thread(target=self._monitor, daemon=True).start()
        
        def stop(self):
            self.monitoring = False
            return self.max_memory
        
        def _monitor(self):
            while self.monitoring:
                current_memory = self.process.memory_info().rss
                self.max_memory = max(self.max_memory, current_memory)
                time.sleep(0.1)
    
    return MemoryMonitor()

@pytest.fixture
def stress_test_config():
    """Configuration for stress testing."""
    return {
        "concurrent_users": [1, 5, 10, 25, 50],
        "ramp_up_time": 10,  # seconds
        "test_duration": 60,  # seconds
        "think_time": 1,  # seconds between requests
        "error_threshold": 0.05,  # 5% acceptable error rate
    }

@pytest.fixture(scope="session")
def performance_results_collector():
    """Collect performance test results for reporting."""
    results = {
        "test_results": [],
        "system_metrics": [],
        "thresholds_exceeded": []
    }
    return results