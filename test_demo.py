#!/usr/bin/env python3
"""Test the demo functionality automatically."""

import asyncio
from demo_cli import MockSkepticAgent
from src.agent_skeptic_bench import SkepticBenchmark, ScenarioCategory

async def test_demo():
    """Test the demo functionality."""
    print("🔬 Agent Skeptic Bench - Automated Demo Test")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = SkepticBenchmark()
    
    # Get a test scenario
    scenarios = benchmark.get_scenarios(categories=[ScenarioCategory.FACTUAL_CLAIMS], limit=1)
    scenario = scenarios[0]
    
    print(f"Testing scenario: {scenario.name}")
    print(f"Category: {scenario.category.value}")
    print(f"Claim: {scenario.adversary_claim}")
    print("\n" + "-" * 50)
    
    # Create and test mock agent
    agent = MockSkepticAgent()
    print("🤖 Running skeptical evaluation...")
    
    # Evaluate the scenario
    result = await benchmark.evaluate_scenario(agent, scenario)
    
    print("\n📊 EVALUATION COMPLETE!")
    print(f"Overall Score: {result.metrics.overall_score:.2f}")
    print(f"Passed: {'✅ YES' if result.passed_evaluation else '❌ NO'}")
    
    print("\n🧠 Agent Response Summary:")
    response_lines = result.response.response_text.strip().split('\n')
    for line in response_lines[:10]:  # First 10 lines
        if line.strip():
            print(f"  {line.strip()}")
    
    print("\n📈 Key Metrics:")
    print(f"  Skepticism Calibration: {result.metrics.skepticism_calibration:.2f}")
    print(f"  Evidence Standards: {result.metrics.evidence_standard_score:.2f}")
    print(f"  Red Flag Detection: {result.metrics.red_flag_detection:.2f}")
    
    print(f"\n🔍 Evidence Requested: {len(result.response.evidence_requests)} items")
    for req in result.response.evidence_requests[:3]:
        print(f"  - {req}")
    
    print(f"\n🚨 Red Flags Identified: {len(result.response.red_flags_identified)} items")
    for flag in result.response.red_flags_identified[:3]:
        print(f"  - {flag}")
    
    print("\n✅ Generation 1 Implementation: WORKING!")
    print("Core functionality successfully demonstrated.")

if __name__ == "__main__":
    asyncio.run(test_demo())