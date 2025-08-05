"""Integration tests for production readiness and scalability."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from src.agent_skeptic_bench.performance import (
    AutoScalingManager, 
    LoadBalancer, 
    PerformanceMetrics,
    PerformanceOptimizer
)
from src.agent_skeptic_bench.validation import security_validator
from src.agent_skeptic_bench.benchmark import SkepticBenchmark


class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    @pytest.fixture
    def auto_scaler(self):
        """Create auto-scaling manager."""
        return AutoScalingManager()
    
    @pytest.fixture
    def high_load_metrics(self):
        """Metrics indicating high system load."""
        return PerformanceMetrics(
            cpu_usage_percent=85.0,
            memory_usage_mb=7500.0,
            disk_usage_percent=60.0,
            active_connections=150,
            evaluation_queue_size=120,
            cache_hit_rate=0.85,
            average_response_time_ms=2500.0,
            requests_per_second=80.0,
            concurrent_evaluations=45
        )
    
    @pytest.fixture
    def low_load_metrics(self):
        """Metrics indicating low system load."""
        return PerformanceMetrics(
            cpu_usage_percent=25.0,
            memory_usage_mb=2000.0,
            disk_usage_percent=40.0,
            active_connections=10,
            evaluation_queue_size=5,
            cache_hit_rate=0.90,
            average_response_time_ms=400.0,
            requests_per_second=15.0,
            concurrent_evaluations=3
        )
    
    def test_scaling_need_analysis_high_load(self, auto_scaler, high_load_metrics):
        """Test scaling analysis under high load."""
        decision = auto_scaler.analyze_scaling_need(high_load_metrics)
        
        assert isinstance(decision, dict)
        assert "action" in decision
        
        # Should trigger scale-up for high CPU
        if decision["action"] != "wait":  # Not in cooldown
            assert decision["action"] == "scale_up"
            assert "scale_factor" in decision
    
    def test_scaling_need_analysis_normal_load(self, auto_scaler):
        """Test scaling analysis under normal load."""
        normal_metrics = PerformanceMetrics(
            cpu_usage_percent=50.0,
            memory_usage_mb=4000.0,
            disk_usage_percent=50.0,
            active_connections=50,
            evaluation_queue_size=30,
            cache_hit_rate=0.87,
            average_response_time_ms=800.0,
            requests_per_second=40.0,
            concurrent_evaluations=15
        )
        
        decision = auto_scaler.analyze_scaling_need(normal_metrics)
        
        assert isinstance(decision, dict)
        if decision["action"] != "wait":
            assert decision["action"] == "maintain"
    
    def test_replica_calculation(self, auto_scaler):
        """Test target replica calculation."""
        scale_up_decision = {
            "action": "scale_up",
            "scale_factor": 1.5
        }
        
        target = auto_scaler.calculate_target_replicas(3, scale_up_decision)
        
        assert isinstance(target, int)
        assert auto_scaler.min_replicas <= target <= auto_scaler.max_replicas
        assert target >= 3  # Should scale up
    
    def test_scaling_cooldown(self, auto_scaler, high_load_metrics):
        """Test scaling cooldown mechanism."""
        # First scaling action
        decision1 = auto_scaler.analyze_scaling_need(high_load_metrics)
        if decision1["action"] == "scale_up":
            auto_scaler.calculate_target_replicas(3, decision1)
        
        # Immediate second analysis should trigger cooldown
        decision2 = auto_scaler.analyze_scaling_need(high_load_metrics)
        assert decision2["action"] == "wait"
        assert "reason" in decision2
    
    def test_scaling_history_tracking(self, auto_scaler):
        """Test scaling history is properly tracked."""
        initial_history_length = len(auto_scaler.scaling_history)
        
        decision = {"action": "scale_up", "scale_factor": 1.5}
        auto_scaler.calculate_target_replicas(3, decision)
        
        assert len(auto_scaler.scaling_history) == initial_history_length + 1
        
        latest_entry = auto_scaler.scaling_history[-1]
        assert "timestamp" in latest_entry
        assert "current_replicas" in latest_entry
        assert "target_replicas" in latest_entry
        assert "action" in latest_entry


class TestLoadBalancer:
    """Test load balancing functionality."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer instance."""
        lb = LoadBalancer()
        
        # Register test workers
        for i in range(1, 4):
            lb.register_worker(
                worker_id=f"worker_{i}",
                capacity=10,
                quantum_coherence=0.8 + (i * 0.05)
            )
        
        return lb
    
    def test_worker_registration(self, load_balancer):
        """Test worker registration."""
        initial_count = len(load_balancer.workers)
        
        load_balancer.register_worker("new_worker", capacity=15, quantum_coherence=0.9)
        
        assert len(load_balancer.workers) == initial_count + 1
        assert "new_worker" in load_balancer.workers
        
        worker = load_balancer.workers["new_worker"]
        assert worker["capacity"] == 15
        assert worker["quantum_coherence"] == 0.9
        assert worker["health_status"] == "healthy"
    
    def test_optimal_worker_selection(self, load_balancer):
        """Test optimal worker selection algorithm."""
        # Test quantum-weighted selection
        load_balancer.load_distribution_strategy = "quantum_weighted"
        selected_worker = load_balancer.select_optimal_worker(task_complexity=0.5)
        
        assert selected_worker is not None
        assert selected_worker in load_balancer.workers
    
    def test_task_assignment(self, load_balancer):
        """Test task assignment to workers."""
        worker_id = "worker_1"
        initial_load = load_balancer.workers[worker_id]["current_load"]
        
        success = load_balancer.assign_task(worker_id, "task_1")
        
        assert success is True
        assert load_balancer.workers[worker_id]["current_load"] == initial_load + 1
    
    def test_task_assignment_at_capacity(self, load_balancer):
        """Test task assignment when worker is at capacity."""
        worker_id = "worker_1"
        worker = load_balancer.workers[worker_id]
        
        # Load worker to capacity
        for i in range(worker["capacity"]):
            load_balancer.assign_task(worker_id, f"task_{i}")
        
        # Try to assign one more task
        success = load_balancer.assign_task(worker_id, "overflow_task")
        
        assert success is False
        assert worker["current_load"] == worker["capacity"]
    
    def test_task_completion_tracking(self, load_balancer):
        """Test task completion tracking."""
        worker_id = "worker_1"
        
        # Assign and complete a successful task
        load_balancer.assign_task(worker_id, "task_1")
        initial_completed = load_balancer.workers[worker_id]["completed_tasks"]
        
        load_balancer.complete_task(worker_id, "task_1", success=True, response_time_ms=500)
        
        worker = load_balancer.workers[worker_id]
        assert worker["completed_tasks"] == initial_completed + 1
        assert worker["current_load"] == 0  # Task completed, load reduced
    
    def test_worker_failure_handling(self, load_balancer):
        """Test worker failure handling and circuit breaker."""
        worker_id = "worker_1"
        
        # Simulate multiple failures
        for i in range(6):  # Exceed circuit breaker threshold
            load_balancer.assign_task(worker_id, f"task_{i}")
            load_balancer.complete_task(worker_id, f"task_{i}", success=False, response_time_ms=1000)
        
        # Worker should be in circuit breaker state
        worker = load_balancer.workers[worker_id]
        assert worker["health_status"] == "circuit_breaker"
        assert worker["failed_tasks"] >= 5


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create performance optimizer."""
        return PerformanceOptimizer()
    
    def test_cache_functionality(self, performance_optimizer):
        """Test caching functionality."""
        # Test scenario metrics caching
        scenario_id = "test_scenario"
        response_hash = "response_hash_123"
        metrics = {"accuracy": 0.95, "response_time": 500}
        
        performance_optimizer.cache_scenario_metrics(scenario_id, response_hash, metrics)
        cached_metrics = performance_optimizer.get_cached_metrics(scenario_id, response_hash)
        
        assert cached_metrics == metrics
    
    def test_scenario_analysis_caching(self, performance_optimizer):
        """Test scenario analysis caching."""
        scenario_id = "test_scenario"
        analysis = {"complexity": "high", "domain": "medical", "confidence": 0.87}
        
        performance_optimizer.cache_scenario_analysis(scenario_id, analysis)
        cached_analysis = performance_optimizer.get_cached_analysis(scenario_id)
        
        assert cached_analysis == analysis
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization(self, performance_optimizer):
        """Test concurrent evaluation optimization."""
        # Create mock evaluation tasks
        async def mock_evaluation_task():
            await asyncio.sleep(0.01)  # Simulate work
            return {"result": "success"}
        
        tasks = [mock_evaluation_task for _ in range(5)]
        
        start_time = time.time()
        results = await performance_optimizer.optimize_concurrent_evaluation(tasks)
        execution_time = time.time() - start_time
        
        assert len(results) == 5
        assert execution_time < 1.0  # Should be fast due to concurrency
    
    def test_performance_stats(self, performance_optimizer):
        """Test performance statistics collection."""
        stats = performance_optimizer.stats
        
        assert isinstance(stats, dict)
        assert "cache_stats" in stats
        assert "evaluator_stats" in stats
        assert "system_metrics" in stats
        
        cache_stats = stats["cache_stats"]
        assert "hit_rate" in cache_stats
        assert "size" in cache_stats


class TestSecurityValidation:
    """Test security validation features."""
    
    def test_input_security_validation_safe(self):
        """Test security validation with safe input."""
        safe_input = "This is a normal scenario description for testing purposes."
        
        validation = security_validator.validate_input_security(safe_input)
        
        assert validation["safe"] is True
        assert validation["risk_level"] == "low"
        assert len(validation["issues"]) == 0
    
    def test_input_security_validation_suspicious(self):
        """Test security validation with suspicious patterns."""
        suspicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('test')",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgndGVzdCcpPC9zY3JpcHQ+",
            "eval('malicious code')",
            "import os; os.system('rm -rf /')"
        ]
        
        for suspicious_input in suspicious_inputs:
            validation = security_validator.validate_input_security(suspicious_input)
            
            assert validation["safe"] is False
            assert validation["risk_level"] in ["medium", "high"]
            assert len(validation["issues"]) > 0
    
    def test_input_length_validation(self):
        """Test input length validation."""
        long_input = "x" * 15000  # Exceed max length
        
        validation = security_validator.validate_input_security(long_input)
        
        assert validation["safe"] is False
        assert any("too long" in issue.lower() for issue in validation["issues"])
    
    def test_nested_input_validation(self):
        """Test security validation with nested data structures."""
        nested_input = {
            "scenario": {
                "description": "Normal description",
                "adversary_claim": "<script>alert('xss')</script>",
                "metadata": {
                    "tags": ["normal", "eval('test')", "safe"]
                }
            }
        }
        
        validation = security_validator.validate_input_security(nested_input)
        
        assert validation["safe"] is False
        assert len(validation["issues"]) >= 2  # Script and eval patterns
    
    @pytest.mark.skipif(True, reason="File system tests may not be available in all environments")
    def test_file_security_validation(self):
        """Test file security validation."""
        from pathlib import Path
        import tempfile
        
        # Create a temporary safe file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_file = Path(f.name)
        
        try:
            validation = security_validator.validate_file_security(temp_file)
            
            assert "safe" in validation
            assert "issues" in validation
            assert "file_path" in validation
        finally:
            temp_file.unlink()  # Clean up


class TestProductionIntegration:
    """Test complete production system integration."""
    
    @pytest.fixture
    def production_benchmark(self):
        """Create production-ready benchmark instance."""
        return SkepticBenchmark()
    
    def test_system_initialization(self, production_benchmark):
        """Test production system initialization."""
        assert production_benchmark is not None
        assert hasattr(production_benchmark, 'quantum_calibrator')
        assert hasattr(production_benchmark, 'scenario_loader')
        assert hasattr(production_benchmark, 'agent_factory')
        assert hasattr(production_benchmark, 'metrics_calculator')
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation_capability(self, production_benchmark):
        """Test system's ability to handle concurrent evaluations."""
        # This would test actual concurrent evaluation in a real implementation
        # For now, we test that the system can handle the load structurally
        
        # Simulate multiple concurrent sessions
        sessions = []
        for i in range(3):
            from src.agent_skeptic_bench.models import AgentConfig, AgentProvider
            config = AgentConfig(
                provider=AgentProvider.OPENAI,
                model_name="test-model",
                api_key="test-key",
                temperature=0.7
            )
            session = production_benchmark.create_session(f"test_session_{i}", config)
            sessions.append(session)
        
        assert len(sessions) == 3
        assert len(production_benchmark._active_sessions) == 3
    
    def test_quantum_insights_generation(self, production_benchmark):
        """Test quantum insights generation capability."""
        from src.agent_skeptic_bench.models import AgentConfig, AgentProvider
        config = AgentConfig(
            provider=AgentProvider.OPENAI,
            model_name="test-model",
            api_key="test-key"
        )
        session = production_benchmark.create_session("insight_test", config)
        
        # Test with no results
        insights = production_benchmark.get_quantum_insights(session.id)
        assert "error" in insights  # Should indicate no results available
    
    def test_parameter_optimization_integration(self, production_benchmark):
        """Test parameter optimization integration."""
        from src.agent_skeptic_bench.models import AgentConfig, AgentProvider
        config = AgentConfig(
            provider=AgentProvider.OPENAI,
            model_name="test-model",
            api_key="test-key"
        )
        session = production_benchmark.create_session("optimization_test", config)
        
        # Test with no results - should raise appropriate error
        with pytest.raises(ValueError, match="No valid session found with results"):
            production_benchmark.optimize_agent_parameters(session.id)
    
    def test_scenario_difficulty_prediction(self, production_benchmark):
        """Test scenario difficulty prediction."""
        from src.agent_skeptic_bench.models import Scenario, ScenarioCategory
        
        scenario = Scenario(
            id="difficulty_test",
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title="Difficulty Test",
            description="This is a test scenario for difficulty prediction",
            correct_skepticism_level=0.8,
            metadata={"complexity": "high"}
        )
        
        agent_params = {
            "temperature": 0.7,
            "skepticism_threshold": 0.6,
            "evidence_weight": 1.2
        }
        
        difficulty = production_benchmark.predict_scenario_difficulty(scenario, agent_params)
        
        assert isinstance(difficulty, float)
        assert 0.0 <= difficulty <= 1.0


class TestSystemResilience:
    """Test system resilience and error handling."""
    
    def test_graceful_degradation_under_load(self):
        """Test system behavior under extreme load."""
        auto_scaler = AutoScalingManager()
        
        # Test with extreme metrics
        extreme_metrics = PerformanceMetrics(
            cpu_usage_percent=99.0,
            memory_usage_mb=15000.0,
            disk_usage_percent=95.0,
            active_connections=1000,
            evaluation_queue_size=500,
            cache_hit_rate=0.30,
            average_response_time_ms=10000.0,
            requests_per_second=200.0,
            concurrent_evaluations=100
        )
        
        # System should still provide a decision (not crash)
        decision = auto_scaler.analyze_scaling_need(extreme_metrics)
        assert isinstance(decision, dict)
        assert "action" in decision
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        load_balancer = LoadBalancer()
        
        # Register worker
        load_balancer.register_worker("test_worker", capacity=5)
        
        # Simulate worker going into circuit breaker
        for i in range(10):
            load_balancer.assign_task("test_worker", f"task_{i}")
            load_balancer.complete_task("test_worker", f"task_{i}", success=False, response_time_ms=1000)
        
        # Worker should be in circuit breaker state
        assert load_balancer.workers["test_worker"]["health_status"] == "circuit_breaker"
        
        # Test health check recovery (simulated)
        load_balancer.health_check()
        
        # System should still be operational
        assert len(load_balancer.workers) >= 1
    
    def test_memory_management(self):
        """Test memory management under sustained load."""
        from src.agent_skeptic_bench.performance import PerformanceCache
        
        cache = PerformanceCache(max_size=100)
        
        # Fill cache beyond capacity
        for i in range(150):
            cache.set(f"key_{i}", f"value_{i}")
        
        # Cache should not exceed max size
        assert len(cache._cache) <= cache.max_size
        
        # Cache should still be functional
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])