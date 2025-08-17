#!/usr/bin/env python3
"""
Generation 1 Complete Demo - MAKE IT WORK
==========================================

Demonstrates the complete Generation 1 functionality with working evaluation pipeline.
"""

import asyncio
import json
import time
from datetime import datetime

from src.agent_skeptic_bench import (
    SkepticBenchmark,
    MockSkepticAgent,
    AgentConfig,
    AgentProvider,
    EvaluationMetrics,
    EvaluationResult
)
from src.agent_skeptic_bench.evaluation import SkepticismEvaluator


async def run_generation_1_demo():
    """Run complete Generation 1 demonstration."""
    print("🚀 GENERATION 1 DEMO - MAKE IT WORK")
    print("=" * 60)
    print("Testing core functionality with mock agents")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. Initialize benchmark system
    print("📋 Step 1: Initialize Benchmark System")
    benchmark = SkepticBenchmark()
    print(f"✅ Benchmark initialized")
    
    # 2. Create mock agent for testing
    print("\n🤖 Step 2: Create Mock Agent")
    config = AgentConfig(
        provider=AgentProvider.CUSTOM,
        model_name="mock-skeptic-v1",
        api_key="demo-key",
        temperature=0.5
    )
    agent = MockSkepticAgent(config)
    print(f"✅ Mock agent created: {agent.agent_id}")
    
    # 3. Load test scenarios
    print("\n📝 Step 3: Load Evaluation Scenarios")
    scenarios = benchmark.get_scenarios(limit=3)
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario.name} ({scenario.category.value})")
    
    # 4. Run evaluations
    print("\n🧪 Step 4: Run Agent Evaluations")
    evaluator = SkepticismEvaluator()
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n  Evaluating scenario {i}: {scenario.name}")
        
        # Get agent response
        response = await agent.evaluate_claim(scenario)
        
        # Evaluate the response
        metrics = evaluator.evaluate_skepticism(scenario, response)
        
        # Create evaluation result
        result = EvaluationResult(
            task_id=f"gen1_demo_{i}",
            scenario=scenario,
            response=response,
            metrics=metrics,
            analysis={
                "generation": "1_simple",
                "agent_type": "mock",
                "test_mode": True
            }
        )
        
        results.append(result)
        
        # Print results
        print(f"    ✅ Confidence: {response.confidence_level:.2f}")
        print(f"    ✅ Evidence requests: {len(response.evidence_requests)}")
        print(f"    ✅ Red flags identified: {len(response.red_flags_identified)}")
        print(f"    ✅ Overall score: {metrics.overall_score:.3f}")
        print(f"    ✅ Passed evaluation: {result.passed_evaluation}")
    
    # 5. Generate summary report
    print("\n📊 Step 5: Generate Summary Report")
    
    total_scenarios = len(results)
    passed_scenarios = sum(1 for r in results if r.passed_evaluation)
    avg_score = sum(r.metrics.overall_score for r in results) / total_scenarios
    avg_response_time = sum(r.response.response_time_ms for r in results) / total_scenarios
    
    # Calculate detailed metrics
    avg_skepticism = sum(r.metrics.skepticism_calibration for r in results) / total_scenarios
    avg_evidence = sum(r.metrics.evidence_standard_score for r in results) / total_scenarios
    avg_red_flag = sum(r.metrics.red_flag_detection for r in results) / total_scenarios
    avg_reasoning = sum(r.metrics.reasoning_quality for r in results) / total_scenarios
    
    execution_time = time.time() - start_time
    
    print(f"✅ Evaluation completed in {execution_time:.2f}s")
    print(f"✅ Scenarios evaluated: {total_scenarios}")
    print(f"✅ Scenarios passed: {passed_scenarios}/{total_scenarios} ({(passed_scenarios/total_scenarios)*100:.1f}%)")
    print(f"✅ Average overall score: {avg_score:.3f}")
    print(f"✅ Average response time: {avg_response_time:.1f}ms")
    
    print(f"\n📈 Detailed Metrics:")
    print(f"  • Skepticism calibration: {avg_skepticism:.3f}")
    print(f"  • Evidence standards: {avg_evidence:.3f}")
    print(f"  • Red flag detection: {avg_red_flag:.3f}")
    print(f"  • Reasoning quality: {avg_reasoning:.3f}")
    
    # 6. Save results
    print("\n💾 Step 6: Save Results")
    
    results_data = {
        "generation": "1_simple",
        "timestamp": datetime.utcnow().isoformat(),
        "execution_time": execution_time,
        "summary": {
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "pass_rate": passed_scenarios / total_scenarios,
            "average_score": avg_score,
            "average_response_time_ms": avg_response_time
        },
        "detailed_metrics": {
            "skepticism_calibration": avg_skepticism,
            "evidence_standard_score": avg_evidence,
            "red_flag_detection": avg_red_flag,
            "reasoning_quality": avg_reasoning
        },
        "results": [
            {
                "scenario_id": r.scenario.id,
                "scenario_name": r.scenario.name,
                "category": r.scenario.category.value,
                "passed": r.passed_evaluation,
                "overall_score": r.metrics.overall_score,
                "response_time_ms": r.response.response_time_ms
            }
            for r in results
        ]
    }
    
    with open("generation_1_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✅ Results saved to generation_1_results.json")
    
    # 7. Verify Generation 1 success criteria
    print("\n🎯 Step 7: Verify Success Criteria")
    
    success_criteria = {
        "basic_functionality": True,  # All functions executed
        "mock_agent_working": True,   # Mock agent responded
        "evaluation_pipeline": True,  # Evaluation completed
        "results_generated": avg_score > 0,  # Got valid scores
        "reasonable_performance": avg_score > 0.3,  # Minimum performance
    }
    
    all_passed = all(success_criteria.values())
    
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion.replace('_', ' ').title()}")
    
    print(f"\n🏆 GENERATION 1 RESULT: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    if all_passed:
        print("🎉 Generation 1 (MAKE IT WORK) completed successfully!")
        print("🚀 Ready to proceed to Generation 2 (MAKE IT ROBUST)")
    else:
        print("⚠️  Generation 1 has issues that need to be resolved")
    
    return {
        "success": all_passed,
        "execution_time": execution_time,
        "results": results_data,
        "criteria": success_criteria
    }


if __name__ == "__main__":
    # Run the demonstration
    result = asyncio.run(run_generation_1_demo())
    
    # Exit with appropriate code
    exit(0 if result["success"] else 1)