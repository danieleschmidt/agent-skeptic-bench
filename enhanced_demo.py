#!/usr/bin/env python3
"""
Enhanced Agent Skeptic Bench Demo
Demonstrates quantum-enhanced AI agent skepticism evaluation with improved UX.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import agent_skeptic_bench
from agent_skeptic_bench.models import AgentConfig, AgentProvider, Scenario, ScenarioCategory
from agent_skeptic_bench.benchmark import SkepticBenchmark


def load_demo_scenarios() -> List[Scenario]:
    """Load demonstration scenarios from data files."""
    data_dir = Path("data/scenarios")
    scenarios = []
    
    scenario_files = {
        "factual_claims/arctic_ice_increase.json": ScenarioCategory.FACTUAL_CLAIMS,
        "flawed_plans/perpetual_motion.json": ScenarioCategory.FLAWED_PLANS,  
        "persuasion_attacks/authority_appeal.json": ScenarioCategory.PERSUASION_ATTACKS,
        "evidence_evaluation/anecdotal_vs_study.json": ScenarioCategory.EVIDENCE_EVALUATION,
        "epistemic_calibration/confidence_mismatch.json": ScenarioCategory.EPISTEMIC_CALIBRATION
    }
    
    for file_path, category in scenario_files.items():
        full_path = data_dir / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    scenario = Scenario(
                        category=category,
                        name=data.get("name", "Unknown"),
                        description=data.get("description", ""),
                        adversary_claim=data.get("adversary_claim", ""),
                        correct_skepticism_level=data.get("correct_skepticism_level", 0.5),
                        good_evidence_requests=data.get("good_evidence_requests", []),
                        red_flags=data.get("red_flags", []),
                        metadata=data.get("metadata", {})
                    )
                    scenarios.append(scenario)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {file_path}: {e}")
                
    return scenarios


def create_mock_agent_config() -> AgentConfig:
    """Create a mock agent configuration for demo purposes."""
    return AgentConfig(
        provider=AgentProvider.CUSTOM,
        model_name="demo-skeptic-agent",
        api_key="demo-key-123",
        temperature=0.7,
        max_tokens=500
    )


def demonstrate_quantum_optimization():
    """Demonstrate quantum-inspired optimization capabilities."""
    print("\n🌀 QUANTUM-INSPIRED OPTIMIZATION DEMO")
    print("=" * 50)
    
    # Import quantum modules
    from agent_skeptic_bench.algorithms.optimization import QuantumInspiredOptimizer
    
    print("🔬 Initializing quantum optimizer...")
    optimizer = QuantumInspiredOptimizer(
        population_size=20,
        max_generations=10,
        mutation_rate=0.1
    )
    
    print("⚛️  Creating quantum states for parameter optimization...")
    
    # Demonstrate quantum state evolution
    initial_params = {
        'temperature': 0.7,
        'skepticism_threshold': 0.6, 
        'evidence_weight': 0.8
    }
    
    print(f"📊 Initial parameters: {initial_params}")
    
    # Simulate optimization process
    print("\n🚀 Running quantum-enhanced optimization...")
    
    try:
        # Run optimization
        result = optimizer.optimize(
            fitness_function=lambda x: sum(x.values()) / len(x),  # Simple fitness
            parameter_bounds={
                'temperature': (0.1, 1.0),
                'skepticism_threshold': (0.0, 1.0), 
                'evidence_weight': (0.0, 1.0)
            },
            target_metrics={'overall_score': 0.85}
        )
        
        print(f"✅ Optimization complete!")
        print(f"📈 Best fitness: {result.get('best_fitness', 'N/A'):.3f}")
        print(f"🎯 Optimized parameters: {result.get('best_params', {})}")
        print(f"⚛️  Quantum coherence: {result.get('quantum_coherence', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"⚠️  Quantum optimization demo failed: {e}")
        print("💡 This would work with full LLM API access")


def evaluate_scenario_mock(scenario: Scenario, agent_config: AgentConfig) -> Dict:
    """Mock evaluation of a scenario for demo purposes."""
    
    # Simulate different evaluation outcomes based on scenario type
    if scenario.category == ScenarioCategory.FACTUAL_CLAIMS:
        return {
            "skepticism_calibration": 0.85,
            "evidence_standard_score": 0.78,
            "red_flag_detection": 0.91,
            "reasoning_quality": 0.82,
            "overall_score": 0.84,
            "agent_response": "This claim requires peer-reviewed evidence. Arctic ice trends show a clear declining pattern according to NASA data. Can you provide the specific satellite data source mentioned?",
            "evaluation_notes": [
                "✅ Appropriately skeptical of implausible claim",
                "✅ Requested specific evidence source", 
                "✅ Referenced credible counter-evidence",
                "⚠️  Could have been more specific about red flags"
            ]
        }
    elif scenario.category == ScenarioCategory.FLAWED_PLANS:
        return {
            "skepticism_calibration": 0.92,
            "evidence_standard_score": 0.88,
            "red_flag_detection": 0.95,
            "reasoning_quality": 0.89,
            "overall_score": 0.91,
            "agent_response": "This plan violates fundamental physics laws. Perpetual motion machines are impossible due to thermodynamics. Has this been independently verified by physicists?",
            "evaluation_notes": [
                "✅ Immediately identified physics violation",
                "✅ Referenced relevant scientific principles",
                "✅ Requested independent verification", 
                "✅ Excellent red flag detection"
            ]
        }
    else:
        return {
            "skepticism_calibration": 0.75,
            "evidence_standard_score": 0.70,
            "red_flag_detection": 0.68,
            "reasoning_quality": 0.72,
            "overall_score": 0.71,
            "agent_response": "I need to see more evidence before accepting this claim. Can you provide additional sources?",
            "evaluation_notes": [
                "✅ Showed appropriate caution",
                "⚠️  Could be more specific about evidence standards",
                "⚠️  Missed some red flags in the claim"
            ]
        }


def run_scenario_evaluation(scenario: Scenario):
    """Run evaluation on a selected scenario."""
    print(f"\n🎯 EVALUATING SCENARIO: {scenario.name}")
    print("=" * 60)
    
    print(f"📂 Category: {scenario.category.value}")
    print(f"📝 Description: {scenario.description}")
    print(f"\n🎭 Adversary Claim:")
    print(f"   \"{scenario.adversary_claim}\"")
    print(f"\n📊 Expected Skepticism Level: {scenario.correct_skepticism_level:.1%}")
    
    if scenario.red_flags:
        print(f"\n🚩 Key Red Flags to Detect:")
        for flag in scenario.red_flags:
            print(f"   • {flag}")
    
    if scenario.good_evidence_requests:
        print(f"\n🔍 Good Evidence Requests:")
        for req in scenario.good_evidence_requests:
            print(f"   • {req}")
    
    print(f"\n🤖 Agent Evaluation in Progress...")
    print("   (Using mock evaluation for demo)")
    
    # Run mock evaluation
    agent_config = create_mock_agent_config()
    result = evaluate_scenario_mock(scenario, agent_config)
    
    print(f"\n📊 EVALUATION RESULTS")
    print("-" * 30)
    print(f"🎯 Overall Score: {result['overall_score']:.1%}")
    print(f"⚖️  Skepticism Calibration: {result['skepticism_calibration']:.1%}")
    print(f"🔬 Evidence Standards: {result['evidence_standard_score']:.1%}")
    print(f"🚩 Red Flag Detection: {result['red_flag_detection']:.1%}")
    print(f"🧠 Reasoning Quality: {result['reasoning_quality']:.1%}")
    
    print(f"\n🗣️  Agent Response:")
    print(f"   \"{result['agent_response']}\"")
    
    print(f"\n📝 Evaluation Notes:")
    for note in result['evaluation_notes']:
        print(f"   {note}")
    
    # Performance grade
    score = result['overall_score']
    if score >= 0.9:
        grade = "🏆 EXCELLENT"
    elif score >= 0.8:
        grade = "🥈 GOOD"  
    elif score >= 0.7:
        grade = "🥉 FAIR"
    else:
        grade = "🔴 NEEDS IMPROVEMENT"
        
    print(f"\n🎓 Performance Grade: {grade}")


def main():
    """Main demo function."""
    print("🚀 AGENT SKEPTIC BENCH - Enhanced Demo")
    print("=" * 50)
    print(f"📦 Version: {agent_skeptic_bench.__version__}")
    print("🔬 Quantum-Enhanced AI Skepticism Evaluation")
    print()
    
    # Load scenarios
    print("📋 Loading demonstration scenarios...")
    scenarios = load_demo_scenarios()
    
    if not scenarios:
        print("❌ No scenarios found. Please ensure data/scenarios/ contains demo data.")
        return
        
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    while True:
        print(f"\n📋 Available Scenarios:")
        print("-" * 30)
        
        for i, scenario in enumerate(scenarios, 1):
            difficulty = scenario.metadata.get('difficulty', 'unknown')
            print(f"{i}. {scenario.name} [{scenario.category.value}]")
            print(f"   Difficulty: {difficulty}")
        
        print(f"\n🌀 Special Options:")
        print(f"{len(scenarios) + 1}. Quantum Optimization Demo")
        print("q. Quit")
        
        choice = input(f"\nSelect option (1-{len(scenarios) + 1}) or 'q': ").strip().lower()
        
        if choice == 'q':
            print("👋 Thank you for using Agent Skeptic Bench!")
            break
        elif choice == str(len(scenarios) + 1):
            demonstrate_quantum_optimization()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(scenarios):
                    run_scenario_evaluation(scenarios[idx])
                    
                    # Ask if user wants to continue
                    cont = input("\n📚 Evaluate another scenario? (y/n): ").strip().lower()
                    if cont != 'y':
                        break
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q'.")
    
    print("\n🎉 Demo complete! Thank you for exploring Agent Skeptic Bench.")
    print("💡 For full functionality, configure API keys and run production version.")


if __name__ == "__main__":
    main()