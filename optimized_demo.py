#!/usr/bin/env python3
"""
Optimized demo showing Generation 3 performance enhancements.
"""

import asyncio
import time
from typing import List
from src.agent_skeptic_bench import SkepticBenchmark, ScenarioCategory
from src.agent_skeptic_bench.performance import get_performance_optimizer
from src.agent_skeptic_bench.logging_config import setup_logging, get_evaluation_logger
from test_demo import MockSkepticAgent


async def run_optimized_demo():
    """Run optimized performance demo."""
    print("🚀 Agent Skeptic Bench - Generation 3 Optimized Demo")
    print("=" * 60)
    
    # Setup enhanced logging
    setup_logging(log_level="INFO", enable_structured=False, enable_file=False)
    eval_logger = get_evaluation_logger()
    
    # Initialize benchmark and performance optimizer
    benchmark = SkepticBenchmark()
    optimizer = get_performance_optimizer()
    
    print("📊 Loading scenarios and initializing performance optimizer...")
    scenarios = benchmark.get_scenarios()
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    # Create optimized agent
    agent = MockSkepticAgent("optimized_agent")
    
    print("\n🔧 Performance Optimization Demo:")
    print("-" * 40)
    
    # Test 1: Single evaluation with caching
    print("\n1️⃣ Single Evaluation with Performance Metrics")
    scenario = scenarios[0]
    
    start_time = time.time()
    result = await benchmark.evaluate_scenario(agent, scenario, timeout=30.0)
    evaluation_time = (time.time() - start_time) * 1000
    
    print(f"   Scenario: {scenario.name}")
    print(f"   Evaluation Time: {evaluation_time:.1f}ms")
    print(f"   Overall Score: {result.metrics.overall_score:.3f}")
    print(f"   Passed: {'✅' if result.passed_evaluation else '❌'}")
    
    # Test 2: Concurrent batch evaluation
    print("\n2️⃣ Concurrent Batch Evaluation")
    batch_scenarios = scenarios[:3]
    
    # Create evaluation tasks
    evaluation_tasks = []
    for i, scen in enumerate(batch_scenarios):
        task = lambda s=scen: benchmark.evaluate_scenario(agent, s, timeout=30.0)
        evaluation_tasks.append(task)
    
    start_time = time.time()
    batch_results = await optimizer.optimize_concurrent_evaluation(
        evaluation_tasks, 
        target_latency_ms=2000
    )
    batch_time = (time.time() - start_time) * 1000
    
    successful_results = [r for r in batch_results if not isinstance(r, Exception)]
    failed_results = [r for r in batch_results if isinstance(r, Exception)]
    
    print(f"   Scenarios Evaluated: {len(batch_scenarios)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(failed_results)}")
    print(f"   Total Time: {batch_time:.1f}ms")
    print(f"   Average per Scenario: {batch_time/len(batch_scenarios):.1f}ms")
    
    if successful_results:
        avg_score = sum(r.metrics.overall_score for r in successful_results) / len(successful_results)
        pass_rate = sum(1 for r in successful_results if r.passed_evaluation) / len(successful_results)
        print(f"   Average Score: {avg_score:.3f}")
        print(f"   Pass Rate: {pass_rate:.1%}")
    
    # Test 3: Performance caching demonstration
    print("\n3️⃣ Performance Caching Test")
    test_scenario = scenarios[1]
    
    # First evaluation (cache miss)
    start_time = time.time()
    result1 = await benchmark.evaluate_scenario(agent, test_scenario)
    first_time = (time.time() - start_time) * 1000
    
    # Second evaluation of same scenario (potential cache hit)
    start_time = time.time()
    result2 = await benchmark.evaluate_scenario(agent, test_scenario)
    second_time = (time.time() - start_time) * 1000
    
    print(f"   First Evaluation: {first_time:.1f}ms")
    print(f"   Second Evaluation: {second_time:.1f}ms")
    
    if second_time < first_time * 0.8:
        print(f"   🎯 Cache Hit! {((first_time - second_time) / first_time * 100):.1f}% faster")
    else:
        print(f"   ℹ️  Cache Miss - similar performance")
    
    # Test 4: System performance monitoring
    print("\n4️⃣ System Performance Monitoring")
    stats = optimizer.stats
    
    print(f"   Cache Hit Rate: {stats['cache_stats']['hit_rate']:.1%}")
    print(f"   Cache Utilization: {stats['cache_stats']['utilization']:.1%}")
    print(f"   CPU Usage: {stats['system_metrics']['cpu_usage_percent']:.1f}%")
    print(f"   Memory Usage: {stats['system_metrics']['memory_usage_mb']:.1f} MB")
    print(f"   Active Evaluations: {stats['system_metrics']['concurrent_evaluations']}")
    
    # Test 5: Stress test with all scenarios
    print("\n5️⃣ Stress Test - All Scenarios")
    
    all_tasks = []
    for scenario in scenarios:
        task = lambda s=scenario: benchmark.evaluate_scenario(agent, s, timeout=30.0)
        all_tasks.append(task)
    
    stress_start = time.time()
    stress_results = await optimizer.optimize_concurrent_evaluation(
        all_tasks,
        target_latency_ms=3000
    )
    stress_time = (time.time() - stress_start) * 1000
    
    stress_successful = [r for r in stress_results if not isinstance(r, Exception)]
    stress_failed = [r for r in stress_results if isinstance(r, Exception)]
    
    print(f"   Total Scenarios: {len(scenarios)}")
    print(f"   Successful: {len(stress_successful)}")  
    print(f"   Failed: {len(stress_failed)}")
    print(f"   Total Time: {stress_time:.1f}ms")
    print(f"   Throughput: {len(scenarios) / (stress_time/1000):.1f} scenarios/second")
    
    if stress_successful:
        stress_avg_score = sum(r.metrics.overall_score for r in stress_successful) / len(stress_successful)
        stress_pass_rate = sum(1 for r in stress_successful if r.passed_evaluation) / len(stress_successful)
        print(f"   Average Score: {stress_avg_score:.3f}")
        print(f"   Pass Rate: {stress_pass_rate:.1%}")
    
    # Final performance summary
    print("\n📊 Final Performance Summary:")
    print("=" * 40)
    final_stats = optimizer.stats
    eval_stats = final_stats['evaluator_stats']
    
    print(f"Completed Evaluations: {eval_stats['completed_evaluations']}")
    print(f"Failed Evaluations: {eval_stats['failed_evaluations']}")
    print(f"Success Rate: {eval_stats['success_rate']:.1%}")
    print(f"Average Evaluation Time: {eval_stats['average_evaluation_time_s']:.2f}s")
    print(f"Evaluations per Second: {eval_stats['evaluations_per_second']:.1f}")
    print(f"Cache Hit Rate: {final_stats['cache_stats']['hit_rate']:.1%}")
    
    # Cleanup
    await optimizer.shutdown()
    
    print("\n✅ Generation 3 Optimization Demo Complete!")
    print("Key improvements demonstrated:")
    print("  • Concurrent evaluation with dynamic concurrency")
    print("  • Performance caching for repeated evaluations")
    print("  • System resource monitoring and optimization") 
    print("  • High-throughput batch processing")
    print("  • Adaptive performance tuning")


if __name__ == "__main__":
    asyncio.run(run_optimized_demo())