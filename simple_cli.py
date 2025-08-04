#!/usr/bin/env python3
"""
Simple CLI for Agent Skeptic Bench - Generation 1

Usage: python3 simple_cli.py [command] [options]
"""

import sys
import asyncio
from src.agent_skeptic_bench import SkepticBenchmark, ScenarioCategory
from test_demo import MockSkepticAgent

def show_help():
    """Show help information."""
    print("""
Agent Skeptic Bench - Simple CLI

Commands:
  list              List all available scenarios
  categories        Show scenario categories  
  evaluate <id>     Evaluate a specific scenario
  demo              Run interactive demo
  help              Show this help

Examples:
  python3 simple_cli.py list
  python3 simple_cli.py evaluate factual_001_arctic_ice
  python3 simple_cli.py demo
""")

def list_scenarios():
    """List all available scenarios."""
    benchmark = SkepticBenchmark()
    scenarios = benchmark.get_scenarios()
    
    print(f"\n📋 Available Scenarios ({len(scenarios)} total):")
    print("=" * 60)
    
    by_category = {}
    for scenario in scenarios:
        cat = scenario.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(scenario)
    
    for category, scenarios in by_category.items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(scenarios)} scenarios):")
        for scenario in scenarios:
            difficulty = scenario.metadata.get('difficulty', 'unknown')
            print(f"  • {scenario.id}: {scenario.name} [{difficulty}]")

def show_categories():
    """Show scenario categories."""
    print("\n📂 Scenario Categories:")
    print("=" * 30)
    for category in ScenarioCategory:
        print(f"  • {category.value}")

async def evaluate_scenario(scenario_id: str):
    """Evaluate a specific scenario."""
    benchmark = SkepticBenchmark()
    scenario = benchmark.get_scenario(scenario_id)
    
    if not scenario:
        print(f"❌ Scenario '{scenario_id}' not found.")
        print("Use 'list' command to see available scenarios.")
        return
    
    print(f"\n🎯 Evaluating: {scenario.name}")
    print(f"Category: {scenario.category.value}")
    print(f"Difficulty: {scenario.metadata.get('difficulty', 'unknown')}")
    print("\nClaim:")
    print(f'"{scenario.adversary_claim}"')
    print("\n" + "─" * 50)
    
    agent = MockSkepticAgent()
    print("🤖 Analyzing with skeptical agent...")
    
    result = await benchmark.evaluate_scenario(agent, scenario)
    
    print(f"\n📊 Results:")
    print(f"Overall Score: {result.metrics.overall_score:.2f}")
    print(f"Pass/Fail: {'✅ PASS' if result.passed_evaluation else '❌ FAIL'}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Skepticism Calibration: {result.metrics.skepticism_calibration:.2f}")
    print(f"  Evidence Standards: {result.metrics.evidence_standard_score:.2f}")
    print(f"  Red Flag Detection: {result.metrics.red_flag_detection:.2f}")
    print(f"  Reasoning Quality: {result.metrics.reasoning_quality:.2f}")
    
    print(f"\nEvidence Requested ({len(result.response.evidence_requests)}):")
    for req in result.response.evidence_requests[:5]:
        print(f"  • {req}")
    
    print(f"\nRed Flags Identified ({len(result.response.red_flags_identified)}):")
    for flag in result.response.red_flags_identified[:5]:
        print(f"  • {flag}")

def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help" or command == "--help" or command == "-h":
        show_help()
    elif command == "list":
        list_scenarios()
    elif command == "categories":
        show_categories()
    elif command == "evaluate":
        if len(sys.argv) < 3:
            print("❌ Please specify a scenario ID")
            print("Use 'list' command to see available scenarios.")
            return
        scenario_id = sys.argv[2]
        asyncio.run(evaluate_scenario(scenario_id))
    elif command == "demo":
        print("🚀 Starting interactive demo...")
        from demo_cli import run_demo
        asyncio.run(run_demo())
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()