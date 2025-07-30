# Load testing scenarios for specific Agent Skeptic Bench use cases
import pytest
import time
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class LoadTestResult:
    """Result from a load test scenario."""
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    duration: float

class TestRealisticLoadScenarios:
    """Test realistic load scenarios for the benchmark."""

    @pytest.mark.slow
    def test_research_paper_evaluation_load(self, mock_model):
        """Simulate load from researchers evaluating their models."""
        
        # Simulate research scenario: multiple researchers testing models
        researchers = 5
        models_per_researcher = 3
        scenarios_per_model = 20
        
        def researcher_workflow(researcher_id: int):
            """Simulate a researcher's evaluation workflow."""
            results = []
            
            for model_id in range(models_per_researcher):
                model_results = []
                
                for scenario_id in range(scenarios_per_model):
                    start_time = time.time()
                    
                    try:
                        # Simulate model evaluation on a scenario
                        prompt = f"Researcher {researcher_id}, Model {model_id}, Scenario {scenario_id}"
                        response = mock_model.generate(prompt)
                        
                        # Simulate skepticism evaluation
                        time.sleep(random.uniform(0.05, 0.15))  # Variable processing time
                        
                        execution_time = time.time() - start_time
                        model_results.append({
                            "scenario_id": scenario_id,
                            "execution_time": execution_time,
                            "success": True
                        })
                        
                        # Simulate researcher think time
                        time.sleep(random.uniform(0.5, 2.0))
                        
                    except Exception as e:
                        execution_time = time.time() - start_time
                        model_results.append({
                            "scenario_id": scenario_id,
                            "execution_time": execution_time,
                            "success": False,
                            "error": str(e)
                        })
                
                results.append({
                    "model_id": model_id,
                    "scenarios": model_results
                })
            
            return {
                "researcher_id": researcher_id,
                "models": results
            }
        
        # Execute concurrent researcher workflows
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=researchers) as executor:
            futures = [executor.submit(researcher_workflow, i) for i in range(researchers)]
            researcher_results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        # Analyze results
        total_requests = researchers * models_per_researcher * scenarios_per_model
        successful_requests = 0
        response_times = []
        
        for researcher in researcher_results:
            for model in researcher["models"]:
                for scenario in model["scenarios"]:
                    if scenario["success"]:
                        successful_requests += 1
                    response_times.append(scenario["execution_time"])
        
        result = LoadTestResult(
            scenario_name="research_paper_evaluation",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_requests - successful_requests,
            average_response_time=sum(response_times) / len(response_times),
            max_response_time=max(response_times),
            min_response_time=min(response_times),
            requests_per_second=total_requests / total_duration,
            error_rate=1 - (successful_requests / total_requests),
            duration=total_duration
        )
        
        # Assertions
        assert result.error_rate < 0.05  # Less than 5% error rate
        assert result.requests_per_second > 1  # At least 1 request per second
        assert result.average_response_time < 2.0  # Average under 2 seconds
        
        return result

    @pytest.mark.slow 
    def test_continuous_monitoring_load(self, mock_model):
        """Simulate continuous monitoring of model performance."""
        
        # Simulate 24/7 monitoring scenario
        monitoring_duration = 60  # 1 minute simulation of continuous monitoring
        check_interval = 5  # Check every 5 seconds
        models_to_monitor = 3
        
        def monitoring_worker(model_id: int, stop_event):
            """Continuous monitoring worker for a single model."""
            checks_performed = 0
            check_results = []
            
            while not stop_event.is_set():
                start_time = time.time()
                
                try:
                    # Perform health check scenario
                    health_check_prompt = f"Model {model_id} health check at {time.time()}"
                    response = mock_model.generate(health_check_prompt)
                    
                    execution_time = time.time() - start_time
                    check_results.append({
                        "timestamp": start_time,
                        "execution_time": execution_time,
                        "success": True,
                        "model_id": model_id
                    })
                    checks_performed += 1
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    check_results.append({
                        "timestamp": start_time,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e),
                        "model_id": model_id
                    })
                
                # Wait for next check interval
                time.sleep(check_interval)
            
            return {
                "model_id": model_id,
                "checks_performed": checks_performed,
                "results": check_results
            }
        
        # Start monitoring workers
        import threading
        stop_event = threading.Event()
        
        with ThreadPoolExecutor(max_workers=models_to_monitor) as executor:
            futures = [
                executor.submit(monitoring_worker, i, stop_event) 
                for i in range(models_to_monitor)
            ]
            
            # Let monitoring run for specified duration
            time.sleep(monitoring_duration)
            stop_event.set()
            
            # Collect results
            monitoring_results = [future.result() for future in futures]
        
        # Analyze monitoring results
        total_checks = sum(r["checks_performed"] for r in monitoring_results)
        all_results = []
        for monitor in monitoring_results:
            all_results.extend(monitor["results"])
        
        successful_checks = sum(1 for r in all_results if r["success"])
        response_times = [r["execution_time"] for r in all_results]
        
        result = LoadTestResult(
            scenario_name="continuous_monitoring",
            total_requests=total_checks,
            successful_requests=successful_checks,
            failed_requests=total_checks - successful_checks,
            average_response_time=sum(response_times) / len(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            requests_per_second=total_checks / monitoring_duration,
            error_rate=1 - (successful_checks / total_checks) if total_checks > 0 else 0,
            duration=monitoring_duration
        )
        
        # Assertions for monitoring scenario
        assert result.error_rate < 0.02  # Very low error rate for monitoring
        assert result.total_requests >= (monitoring_duration // check_interval) * models_to_monitor * 0.8  # Account for timing variations
        
        return result

    @pytest.mark.benchmark
    def test_batch_evaluation_load(self, mock_model, large_scenario_dataset):
        """Test load from batch evaluation scenarios."""
        
        batch_sizes = [10, 25, 50, 100]
        batch_results = []
        
        for batch_size in batch_sizes:
            scenarios = large_scenario_dataset[:batch_size]
            
            def process_batch(scenarios_batch):
                results = []
                start_time = time.time()
                
                for scenario in scenarios_batch:
                    scenario_start = time.time()
                    
                    try:
                        response = mock_model.generate(scenario["adversary_claim"])
                        scenario_time = time.time() - scenario_start
                        
                        results.append({
                            "scenario_id": scenario["id"],
                            "execution_time": scenario_time,
                            "success": True,
                            "response": response
                        })
                        
                    except Exception as e:
                        scenario_time = time.time() - scenario_start
                        results.append({
                            "scenario_id": scenario["id"],
                            "execution_time": scenario_time,
                            "success": False,
                            "error": str(e)
                        })
                
                total_time = time.time() - start_time
                return results, total_time
            
            batch_result, duration = process_batch(scenarios)
            
            successful = sum(1 for r in batch_result if r["success"])
            response_times = [r["execution_time"] for r in batch_result]
            
            result = LoadTestResult(
                scenario_name=f"batch_evaluation_{batch_size}",
                total_requests=batch_size,
                successful_requests=successful,
                failed_requests=batch_size - successful,
                average_response_time=sum(response_times) / len(response_times),
                max_response_time=max(response_times),
                min_response_time=min(response_times),
                requests_per_second=batch_size / duration,
                error_rate=1 - (successful / batch_size),
                duration=duration
            )
            
            batch_results.append(result)
            
            # Per-batch assertions
            assert result.error_rate < 0.05
            assert result.average_response_time < 1.0
        
        # Overall batch processing should scale reasonably
        throughputs = [r.requests_per_second for r in batch_results]
        assert max(throughputs) > min(throughputs) * 1.2  # Some scalability improvement
        
        return batch_results

    @pytest.mark.slow
    def test_peak_usage_simulation(self, mock_model):
        """Simulate peak usage periods (e.g., conference deadlines)."""
        
        # Simulate conference deadline scenario
        peak_users = 20
        scenarios_per_user = 15
        time_pressure_factor = 0.1  # Reduced think time during peak
        
        def peak_user_behavior(user_id: int):
            """Simulate user behavior during peak periods."""
            user_results = []
            
            for scenario_num in range(scenarios_per_user):
                start_time = time.time()
                
                try:
                    # Users may retry scenarios more during peak times
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            prompt = f"Peak user {user_id}, scenario {scenario_num}, attempt {attempt}"
                            response = mock_model.generate(prompt)
                            break
                        except Exception:
                            if attempt == max_retries - 1:
                                raise
                            time.sleep(0.1)  # Brief retry delay
                    
                    execution_time = time.time() - start_time
                    user_results.append({
                        "scenario_num": scenario_num,
                        "execution_time": execution_time,
                        "success": True
                    })
                    
                    # Reduced think time during peak usage
                    time.sleep(random.uniform(0.1, 0.5) * time_pressure_factor)
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    user_results.append({
                        "scenario_num": scenario_num,
                        "execution_time": execution_time,
                        "success": False,
                        "error": str(e)
                    })
                    
                    # Frustrated users might retry more quickly
                    time.sleep(0.1)
            
            return {
                "user_id": user_id,
                "results": user_results
            }
        
        # Execute peak load test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=peak_users) as executor:
            futures = [executor.submit(peak_user_behavior, i) for i in range(peak_users)]
            user_results = [future.result() for future in as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        # Aggregate results
        all_scenario_results = []
        for user in user_results:
            all_scenario_results.extend(user["results"])
        
        total_requests = len(all_scenario_results)
        successful_requests = sum(1 for r in all_scenario_results if r["success"])
        response_times = [r["execution_time"] for r in all_scenario_results]
        
        result = LoadTestResult(
            scenario_name="peak_usage_simulation",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_requests - successful_requests,
            average_response_time=sum(response_times) / len(response_times),
            max_response_time=max(response_times),
            min_response_time=min(response_times),
            requests_per_second=total_requests / total_duration,
            error_rate=1 - (successful_requests / total_requests),
            duration=total_duration
        )
        
        # Peak usage should still maintain reasonable performance
        assert result.error_rate < 0.10  # Allow higher error rate during peak
        assert result.requests_per_second > 5  # Maintain minimum throughput
        assert result.average_response_time < 3.0  # Allow higher latency during peak
        
        return result

@pytest.mark.slow
def test_load_test_report_generation(tmp_path):
    """Generate a comprehensive load test report."""
    
    # This would typically run all the load tests and compile results
    sample_results = [
        LoadTestResult(
            scenario_name="sample_test",
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            average_response_time=0.25,
            max_response_time=1.2,
            min_response_time=0.1,
            requests_per_second=15.2,
            error_rate=0.02,
            duration=6.58
        )
    ]
    
    # Generate HTML report
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Load Test Report - Agent Skeptic Bench</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .pass {{ color: green; }}
            .warn {{ color: orange; }}
            .fail {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Load Test Report</h1>
        <h2>Test Results Summary</h2>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Total Requests</th>
                <th>Success Rate</th>
                <th>Avg Response Time</th>
                <th>Throughput (req/s)</th>
                <th>Status</th>
            </tr>
    """
    
    for result in sample_results:
        status_class = "pass" if result.error_rate < 0.05 else "fail"
        report_html += f"""
            <tr>
                <td>{result.scenario_name}</td>
                <td>{result.total_requests}</td>
                <td>{(1-result.error_rate)*100:.1f}%</td>
                <td>{result.average_response_time:.3f}s</td>
                <td>{result.requests_per_second:.1f}</td>
                <td class="{status_class}">{'PASS' if result.error_rate < 0.05 else 'FAIL'}</td>
            </tr>
        """
    
    report_html += """
        </table>
    </body>
    </html>
    """
    
    report_path = tmp_path / "load_test_report.html"
    report_path.write_text(report_html)
    
    # Also generate JSON report for programmatic analysis
    json_report = {
        "timestamp": time.time(),
        "results": [asdict(result) for result in sample_results]
    }
    
    json_path = tmp_path / "load_test_results.json"
    json_path.write_text(json.dumps(json_report, indent=2))
    
    assert report_path.exists()
    assert json_path.exists()
    
    return {
        "html_report": str(report_path),
        "json_report": str(json_path)
    }