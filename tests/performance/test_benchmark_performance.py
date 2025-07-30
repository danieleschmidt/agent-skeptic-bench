# Performance tests for Agent Skeptic Bench
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import threading

class TestBenchmarkPerformance:
    """Performance tests for benchmark execution."""

    @pytest.mark.benchmark
    def test_single_scenario_execution_time(self, benchmark, mock_model, benchmark_config):
        """Test execution time for a single scenario."""
        
        def run_scenario():
            # Simulate scenario execution
            start_time = time.time()
            result = mock_model.generate("Test scenario prompt")
            execution_time = time.time() - start_time
            
            # Simulate skepticism scoring
            time.sleep(0.05)  # Simulate processing
            skepticism_score = 0.75
            
            return {
                "execution_time": execution_time,
                "skepticism_score": skepticism_score,
                "result": result
            }
        
        # Benchmark the scenario execution
        result = benchmark(run_scenario)
        
        # Assert performance requirements
        assert result["execution_time"] < benchmark_config["acceptable_latency"]
        assert result["skepticism_score"] is not None

    @pytest.mark.benchmark
    def test_batch_scenario_processing(self, benchmark, mock_model, benchmark_config):
        """Test batch processing performance."""
        
        def process_batch(batch_size: int = 10):
            scenarios = [f"Scenario {i}" for i in range(batch_size)]
            results = []
            
            start_time = time.time()
            for scenario in scenarios:
                result = mock_model.generate(scenario)
                results.append(result)
            
            total_time = time.time() - start_time
            return {
                "batch_size": batch_size,
                "total_time": total_time,
                "average_time_per_scenario": total_time / batch_size,
                "results": results
            }
        
        # Benchmark batch processing
        result = benchmark.pedantic(process_batch, kwargs={"batch_size": 50}, rounds=5)
        
        # Performance assertions
        assert result["average_time_per_scenario"] < benchmark_config["acceptable_latency"]
        assert len(result["results"]) == 50

    @pytest.mark.benchmark
    def test_concurrent_scenario_execution(self, mock_model, benchmark_config):
        """Test concurrent execution performance."""
        
        def concurrent_execution(num_threads: int = 5, scenarios_per_thread: int = 10):
            results = []
            execution_times = []
            
            def worker_thread(thread_id: int):
                thread_results = []
                for i in range(scenarios_per_thread):
                    start_time = time.time()
                    result = mock_model.generate(f"Thread {thread_id} scenario {i}")
                    execution_time = time.time() - start_time
                    
                    thread_results.append(result)
                    execution_times.append(execution_time)
                
                return thread_results
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
                
                for future in as_completed(futures):
                    thread_results = future.result()
                    results.extend(thread_results)
            
            total_time = time.time() - start_time
            
            return {
                "total_scenarios": num_threads * scenarios_per_thread,
                "total_time": total_time,
                "average_execution_time": statistics.mean(execution_times),
                "max_execution_time": max(execution_times),
                "throughput": len(results) / total_time
            }
        
        result = concurrent_execution(num_threads=5, scenarios_per_thread=20)
        
        # Performance assertions
        assert result["throughput"] > 10  # scenarios per second
        assert result["average_execution_time"] < benchmark_config["acceptable_latency"]
        assert result["total_scenarios"] == 100

    @pytest.mark.slow
    def test_memory_usage_during_execution(self, memory_monitor, large_scenario_dataset, mock_model, benchmark_config):
        """Test memory usage during large batch processing."""
        
        memory_monitor.start()
        
        try:
            # Process large dataset
            results = []
            for scenario in large_scenario_dataset[:100]:  # Process first 100 scenarios
                result = mock_model.generate(scenario["adversary_claim"])
                results.append({
                    "scenario_id": scenario["id"],
                    "result": result,
                    "skepticism_score": scenario["expected_skepticism"]
                })
        
        finally:
            max_memory = memory_monitor.stop()
        
        # Memory assertions
        assert max_memory < benchmark_config["memory_limit"]
        assert len(results) == 100

    @pytest.mark.benchmark
    def test_model_response_time_distribution(self, mock_model):
        """Test distribution of model response times."""
        
        response_times = []
        num_requests = 100
        
        for i in range(num_requests):
            start_time = time.time()
            mock_model.generate(f"Test prompt {i}")
            response_time = time.time() - start_time
            response_times.append(response_time)
        
        # Calculate statistics
        mean_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))]
        p99_time = sorted(response_times)[int(0.99 * len(response_times))]
        
        # Performance assertions
        assert mean_time < 0.2  # 200ms average
        assert p95_time < 0.3   # 300ms 95th percentile
        assert p99_time < 0.5   # 500ms 99th percentile
        
        # Store results for reporting
        return {
            "mean": mean_time,
            "median": median_time,
            "p95": p95_time,
            "p99": p99_time,
            "total_requests": num_requests
        }

class TestLoadTesting:
    """Load testing scenarios."""

    @pytest.mark.slow
    def test_sustained_load(self, mock_model, stress_test_config):
        """Test sustained load over time."""
        
        def simulate_user_load(user_id: int, duration: int):
            """Simulate a single user's load."""
            start_time = time.time()
            requests_made = 0
            errors = 0
            
            while time.time() - start_time < duration:
                try:
                    request_start = time.time()
                    mock_model.generate(f"User {user_id} request {requests_made}")
                    response_time = time.time() - request_start
                    
                    requests_made += 1
                    
                    # Simulate think time
                    time.sleep(stress_test_config["think_time"])
                    
                except Exception:
                    errors += 1
            
            return {
                "user_id": user_id,
                "requests_made": requests_made,
                "errors": errors,
                "duration": time.time() - start_time
            }
        
        # Test with increasing load
        for num_users in stress_test_config["concurrent_users"]:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(simulate_user_load, i, stress_test_config["test_duration"])
                    for i in range(num_users)
                ]
                
                results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            total_requests = sum(r["requests_made"] for r in results)
            total_errors = sum(r["errors"] for r in results)
            error_rate = total_errors / total_requests if total_requests > 0 else 0
            throughput = total_requests / total_time
            
            # Assertions for each load level
            assert error_rate <= stress_test_config["error_threshold"], f"Error rate {error_rate} exceeds threshold for {num_users} users"
            assert throughput > 0, f"Zero throughput for {num_users} users"
            
            print(f"Load test results for {num_users} users:")
            print(f"  Total requests: {total_requests}")
            print(f"  Error rate: {error_rate:.2%}")
            print(f"  Throughput: {throughput:.2f} req/s")

    @pytest.mark.benchmark
    def test_spike_load_handling(self, mock_model):
        """Test handling of sudden load spikes."""
        
        def spike_test():
            # Normal load
            normal_load_results = []
            for i in range(10):
                start = time.time()
                mock_model.generate(f"Normal load request {i}")
                normal_load_results.append(time.time() - start)
            
            normal_avg = statistics.mean(normal_load_results)
            
            # Spike load
            spike_results = []
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [
                    executor.submit(lambda i=i: mock_model.generate(f"Spike request {i}"))
                    for i in range(50)
                ]
                
                spike_start = time.time()
                for future in as_completed(futures):
                    future.result()
                    spike_results.append(time.time() - spike_start)
            
            spike_duration = max(spike_results)
            
            return {
                "normal_avg_response": normal_avg,
                "spike_duration": spike_duration,
                "spike_requests": len(spike_results)
            }
        
        result = spike_test()
        
        # Spike should complete within reasonable time
        assert result["spike_duration"] < 5.0  # 5 seconds max
        assert result["spike_requests"] == 50

class TestScalabilityMetrics:
    """Test scalability characteristics."""

    @pytest.mark.slow
    def test_throughput_scalability(self, mock_model, performance_results_collector):
        """Test how throughput scales with concurrent requests."""
        
        concurrency_levels = [1, 2, 5, 10, 20]
        scalability_results = []
        
        for concurrency in concurrency_levels:
            def measure_throughput(num_concurrent: int):
                results = []
                
                def worker():
                    start = time.time()
                    mock_model.generate("Scalability test")
                    return time.time() - start
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                    futures = [executor.submit(worker) for _ in range(100)]
                    response_times = [future.result() for future in as_completed(futures)]
                
                total_time = time.time() - start_time
                throughput = 100 / total_time
                avg_response_time = statistics.mean(response_times)
                
                return {
                    "concurrency": num_concurrent,
                    "throughput": throughput,
                    "avg_response_time": avg_response_time,
                    "total_time": total_time
                }
            
            result = measure_throughput(concurrency)
            scalability_results.append(result)
            
            # Store results for analysis
            performance_results_collector["test_results"].append({
                "test_name": "throughput_scalability",
                "concurrency": concurrency,
                "result": result
            })
        
        # Analyze scalability
        throughputs = [r["throughput"] for r in scalability_results]
        
        # Throughput should generally increase with concurrency (up to a point)
        assert throughputs[1] > throughputs[0] * 0.8  # Allow for some variance
        assert max(throughputs) > throughputs[0] * 1.5  # Significant improvement expected
        
        return scalability_results