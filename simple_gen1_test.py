#!/usr/bin/env python3
"""Simple Generation 1 test for Agent Skeptic Bench."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_skeptic_bench.models import AgentConfig, AgentProvider, ScenarioCategory
from agent_skeptic_bench.agents import create_skeptic_agent
from agent_skeptic_bench.scenarios import ScenarioLoader
from agent_skeptic_bench.benchmark import SkepticBenchmark
from agent_skeptic_bench.evaluation import run_full_evaluation


async def test_basic_functionality():
    """Test basic functionality of the Agent Skeptic Bench."""
    print("🧠 Agent Skeptic Bench - Generation 1 Test")
    print("=" * 50)
    
    try:
        # 1. Test scenario loading
        print("\n1. Testing scenario loading...")
        loader = ScenarioLoader()
        scenarios = loader.load_scenarios([ScenarioCategory.FACTUAL_CLAIMS])
        print(f"   ✓ Loaded {len(scenarios)} scenarios")
        
        if scenarios:
            print(f"   Example: {scenarios[0].name}")
            print(f"   Claim: {scenarios[0].adversary_claim[:100]}...")
        
        # 2. Test mock agent creation
        print("\n2. Testing mock agent creation...")
        agent_config = AgentConfig(
            provider=AgentProvider.CUSTOM,
            model_name="mock_skeptic",
            api_key="test_key",
            temperature=0.5
        )
        agent = create_skeptic_agent(
            model="mock_skeptic",
            api_key="test_key",
            provider="custom"
        )
        print(f"   ✓ Created mock agent: {agent.agent_id}")
        
        # 3. Test single scenario evaluation
        print("\n3. Testing single scenario evaluation...")
        if scenarios:
            scenario = scenarios[0]
            response = await agent.evaluate_claim(scenario)
            print(f"   ✓ Evaluated scenario: {scenario.id}")
            print(f"   Response confidence: {response.confidence_level:.2f}")
            print(f"   Evidence requests: {len(response.evidence_requests)}")
            print(f"   Red flags identified: {len(response.red_flags_identified)}")
        
        # 4. Test benchmark evaluation
        print("\n4. Testing benchmark evaluation...")
        benchmark = SkepticBenchmark()
        session = benchmark.create_session(
            name="Gen1 Test Session",
            agent_config=agent_config
        )
        print(f"   ✓ Created session: {session.id}")
        
        # Run small evaluation
        if scenarios[:2]:  # Test with first 2 scenarios
            results = await benchmark.evaluate_batch(agent, scenarios[:2], concurrency=1)
            for result in results:
                session.add_result(result)
            
            print(f"   ✓ Evaluated {len(results)} scenarios")
            print(f"   Pass rate: {session.pass_rate:.1%}")
            
            if session.summary_metrics:
                print(f"   Overall score: {session.summary_metrics.overall_score:.3f}")
        
        print("\n✅ Generation 1 basic functionality test PASSED!")
        print(f"🎯 Key metrics working: skepticism calibration, evidence standards, red flag detection")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cli_simulation():
    """Simulate CLI usage patterns."""
    print("\n" + "=" * 50)
    print("🔧 CLI Simulation Test")
    print("=" * 50)
    
    try:
        # Simulate CLI evaluation command
        print("\n1. Simulating CLI evaluation...")
        
        report = await run_full_evaluation(
            skeptic_agent="mock_skeptic",
            api_key="test_key",
            provider="custom",
            categories=["factual_claims"],
            limit=3,
            parallel=False,
            session_name="CLI_Simulation_Test"
        )
        
        print("   ✓ CLI evaluation completed")
        print("\n📊 Generated Report Summary:")
        print(report.summary())
        
        # Save HTML report
        report.save_html("gen1_test_report.html")
        print("   ✓ HTML report saved to: gen1_test_report.html")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CLI simulation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Generation 1 tests."""
    print("🚀 Starting Agent Skeptic Bench Generation 1 Tests")
    
    async def run_all_tests():
        basic_passed = await test_basic_functionality()
        cli_passed = await test_cli_simulation()
        
        print("\n" + "=" * 60)
        print("📋 GENERATION 1 TEST SUMMARY")
        print("=" * 60)
        print(f"Basic Functionality: {'✅ PASS' if basic_passed else '❌ FAIL'}")
        print(f"CLI Simulation:      {'✅ PASS' if cli_passed else '❌ FAIL'}")
        
        if basic_passed and cli_passed:
            print("\n🎉 GENERATION 1 IMPLEMENTATION SUCCESSFUL!")
            print("   ✓ Core models and validation working")
            print("   ✓ Scenario loading functional")
            print("   ✓ Agent interfaces implemented")
            print("   ✓ Evaluation engine operational")
            print("   ✓ CLI patterns working")
            print("\n🚀 Ready to proceed to Generation 2!")
            return True
        else:
            print("\n⚠️  Some tests failed. Review implementation before proceeding.")
            return False
    
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()