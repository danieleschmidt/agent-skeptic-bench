#!/usr/bin/env python3
"""
Demo of optimized usage metrics system with caching and auto-scaling.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def demo_optimized_usage():
    """Demo optimized usage metrics system."""
    print("🚀 Optimized Usage Metrics Demo")
    print("=" * 50)
    
    # Test core optimization components
    from agent_skeptic_bench.features.usage_cache import MemoryCache, UsageMetricsCache
    from agent_skeptic_bench.features.usage_scaling import AutoScaler, LoadBalancer, PerformanceMonitor
    
    # Test memory cache
    print("\n💾 Testing Memory Cache...")
    cache = MemoryCache(max_size=5, default_ttl=2)
    
    # Test cache operations
    cache.set("test_1", {"data": "value_1"})
    cache.set("test_2", {"data": "value_2"})
    
    result_1 = cache.get("test_1")
    print(f"  ✅ Cache get: {result_1}")
    
    # Test TTL expiration
    await asyncio.sleep(2.5)
    expired_result = cache.get("test_1")
    print(f"  ✅ TTL expiration: {expired_result is None}")
    
    # Test cache stats
    stats = cache.stats()
    print(f"  📊 Cache stats: {stats}")
    
    # Test auto-scaler
    print("\n⚡ Testing Auto-Scaler...")
    scaler = AutoScaler(min_instances=1, max_instances=5)
    
    # Create test metrics
    from agent_skeptic_bench.features.usage_scaling import ScalingMetrics
    
    # High CPU scenario
    high_cpu_metrics = ScalingMetrics(
        cpu_usage=85.0,  # Above threshold
        memory_usage=60.0,
        request_rate=50.0,
        response_time=2.0,
        error_rate=1.0,
        queue_depth=10,
        active_connections=25,
        timestamp=datetime.utcnow()
    )
    
    scaling_decision = scaler.evaluate_scaling(high_cpu_metrics)
    if scaling_decision:
        print(f"  🔼 Scale up decision: {scaling_decision['action']} - {scaling_decision['reason']}")
        print(f"     Instances: {scaling_decision['current_instances']} → {scaling_decision['target_instances']}")
    
    # Test load balancer
    print("\n⚖️ Testing Load Balancer...")
    balancer = LoadBalancer(algorithm="round_robin")
    
    # Add test instances
    balancer.add_instance("instance_1", "localhost:8001", weight=1.0)
    balancer.add_instance("instance_2", "localhost:8002", weight=1.5)
    balancer.add_instance("instance_3", "localhost:8003", weight=1.0)
    
    # Test instance selection
    for i in range(6):
        instance = balancer.get_next_instance()
        if instance:
            # Simulate request
            success = i % 4 != 0  # 25% failure rate for testing
            response_time = 0.5 + (i * 0.1)
            
            balancer.record_request(instance["id"], success=success, response_time=response_time)
            print(f"  🎯 Selected: {instance['id']} (success: {success})")
    
    balancer_stats = balancer.get_balancer_stats()
    print(f"  📊 Balancer stats: {balancer_stats['healthy_instances']}/{balancer_stats['total_instances']} healthy")
    
    # Test performance monitor
    print("\n📈 Testing Performance Monitor...")
    monitor = PerformanceMonitor(monitoring_interval=1)
    
    # Collect some metrics
    for i in range(3):
        metrics = monitor.collect_metrics()
        alerts = monitor.check_alerts(metrics)
        
        print(f"  📊 Metrics {i+1}: CPU={metrics.cpu_usage:.1f}%, Memory={metrics.memory_usage:.1f}%, RT={metrics.response_time:.2f}s")
        
        if alerts:
            for alert in alerts:
                print(f"  🚨 Alert: {alert['message']}")
        
        await asyncio.sleep(0.5)
    
    # Test performance trends
    trends = monitor.get_performance_trends(hours=1)
    if "error" not in trends:
        print(f"  📈 CPU trend: avg={trends['cpu_trend']['avg']:.1f}%, current={trends['cpu_trend']['current']:.1f}%")
    
    # Test integrated optimized manager
    print("\n🎯 Testing Optimized Usage Manager...")
    
    # Import after path setup
    from agent_skeptic_bench.features.optimized_usage_manager import OptimizedUsageManager
    
    # Create manager (without auto-scaling for simplified demo)
    manager = OptimizedUsageManager(
        storage_path="data/optimized_usage",
        enable_caching=True,
        enable_auto_scaling=False  # Simplified for demo
    )
    
    # Start background tasks
    await manager.start_background_tasks()
    
    # Test optimized session creation
    session_result = await manager.create_optimized_session(
        session_id="optimized_session_001",
        user_id="power_user",
        agent_provider="openai",
        model="gpt-4"
    )
    
    print(f"  ✅ Session creation: {session_result}")
    
    # Test optimized evaluation recording
    eval_result = await manager.record_evaluation_optimized(
        session_id="optimized_session_001",
        scenario_id="perf_test_001",
        category="factual_claims",
        duration=1.2,
        score=0.89,
        tokens_used=250
    )
    
    print(f"  ✅ Evaluation recording: {eval_result}")
    
    # Test optimized summary retrieval
    summary_result = await manager.get_usage_summary_optimized(days=7, use_cache=True)
    print(f"  ✅ Summary retrieval: {summary_result['success']}, from_cache: {summary_result.get('from_cache', False)}")
    
    # Test export optimization
    export_result = await manager.export_with_optimization({
        "format": "json",
        "days": 7,
        "summary_only": True
    })
    
    print(f"  ✅ Export optimization: {export_result}")
    
    # Get comprehensive status
    status = await manager.get_comprehensive_status()
    print(f"\n📊 System Status:")
    print(f"  Components: {status['components']}")
    print(f"  Uptime: {status['uptime_seconds']:.1f}s")
    
    if "performance_stats" in status:
        perf_stats = status["performance_stats"]
        print(f"  Operations: {perf_stats['total_operations']}")
        print(f"  Total time: {perf_stats['total_time']:.2f}s")
    
    # Get optimization recommendations
    recommendations = manager.get_optimization_recommendations()
    if recommendations:
        print(f"\n💡 Optimization Recommendations:")
        for rec in recommendations:
            print(f"  - {rec['type']}: {rec['recommendation']}")
    else:
        print(f"\n✅ No optimization recommendations - system performing well!")
    
    # Graceful shutdown
    await manager.shutdown()
    
    print(f"\n🎉 Optimized demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_optimized_usage())