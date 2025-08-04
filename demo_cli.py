#!/usr/bin/env python3
"""
Simple demo CLI for Agent Skeptic Bench - Generation 1 "Make it Work"

This is a basic demonstration of the skeptical evaluation functionality.
"""

import asyncio
import sys
from typing import Optional

from src.agent_skeptic_bench import (
    SkepticBenchmark, 
    create_skeptic_agent,
    AgentConfig,
    AgentProvider,
    ScenarioCategory
)


class MockSkepticAgent:
    """Mock agent for demonstration purposes (no API keys required)."""
    
    def __init__(self, agent_id: str = "demo_skeptic"):
        self.agent_id = agent_id
    
    async def evaluate_claim(self, scenario, context=None):
        """Mock evaluation that demonstrates proper skeptical reasoning."""
        from src.agent_skeptic_bench.models import SkepticResponse
        
        # Simulate thoughtful analysis based on scenario type
        confidence = min(0.95, scenario.correct_skepticism_level + 0.05)
        
        # Generate appropriate skeptical response
        if scenario.category.value == "factual_claims":
            response_text = f"""
I need to evaluate this claim carefully against established scientific evidence.

CLAIM ANALYSIS: {scenario.adversary_claim}

SKEPTICAL ASSESSMENT: 
- This contradicts well-established scientific consensus
- The claim lacks specific data sources and methodologies
- Language suggests potential bias ("narratives pushed by mainstream science")

CONFIDENCE LEVEL: {confidence:.2f} (high skepticism warranted)

EVIDENCE I WOULD REQUEST:
- NASA/NSIDC official satellite data with methodology
- Peer-reviewed climate studies from multiple research groups
- Specific timeframes and measurement criteria
- Independent verification from meteorological institutions

RED FLAGS IDENTIFIED:
- Contradicts scientific consensus without extraordinary evidence
- Vague reference to "recent satellite data" without sources
- Politicization of scientific data
- No acknowledgment of broader climate evidence

REASONING: Extraordinary claims require extraordinary evidence. Climate science 
consensus is based on multiple independent datasets and decades of research.
"""
        
        elif scenario.category.value == "flawed_plans":
            response_text = f"""
This proposal raises immediate red flags regarding fundamental physics.

PLAN ANALYSIS: {scenario.adversary_claim}

SKEPTICAL ASSESSMENT:
- Violates the laws of thermodynamics (conservation of energy)
- "Quantum effects" is used as unexplained handwaving
- Investment component suggests financial motivation over scientific merit

CONFIDENCE LEVEL: {confidence:.2f} (extremely high skepticism)

EVIDENCE I WOULD REQUEST:
- Peer-reviewed physics publications explaining the mechanism
- Independent laboratory verification by qualified physicists
- Detailed energy input/output measurements
- Explanation of how thermodynamic laws are overcome

RED FLAGS IDENTIFIED:
- Violates well-established physics principles
- Investment scheme structure
- Vague quantum terminology without specifics
- No mention of peer review or scientific publication

REASONING: Perpetual motion machines are impossible according to thermodynamics.
Claims to override fundamental physics require extraordinary validation.
"""
        
        else:
            response_text = f"""
I need to carefully evaluate this claim using appropriate skeptical standards.

CLAIM: {scenario.adversary_claim}

My confidence in questioning this claim: {confidence:.2f}

I would request: {', '.join(scenario.good_evidence_requests[:3])}

Key concerns: {', '.join(scenario.red_flags[:3])}
"""
        
        return SkepticResponse(
            agent_id=self.agent_id,
            scenario_id=scenario.id,
            response_text=response_text,
            confidence_level=confidence,
            evidence_requests=scenario.good_evidence_requests[:4],
            red_flags_identified=scenario.red_flags[:4],
            reasoning_steps=[
                f"Analyzed claim against {scenario.category.value} standards",
                "Identified key red flags and evidence gaps",
                "Applied appropriate skepticism level",
                "Formulated evidence-based response"
            ],
            response_time_ms=1500
        )


async def run_demo():
    """Run the demonstration."""
    print("🔬 Agent Skeptic Bench - Demo Mode")
    print("=" * 50)
    print()
    
    # Initialize benchmark
    benchmark = SkepticBenchmark()
    
    # Load available scenarios
    scenarios = benchmark.get_scenarios(limit=5)
    print(f"📋 Loaded {len(scenarios)} demonstration scenarios:")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario.name} [{scenario.category.value}]")
        print(f"   Difficulty: {scenario.metadata.get('difficulty', 'unknown')}")
    
    print("\n" + "=" * 50)
    
    # Let user select a scenario
    while True:
        try:
            choice = input(f"\nSelect scenario (1-{len(scenarios)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                selected_scenario = scenarios[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(scenarios)}")
        except ValueError:
            print("Please enter a valid number or 'q'")
    
    print(f"\n🎯 Selected: {selected_scenario.name}")
    print(f"Category: {selected_scenario.category.value}")
    print(f"Description: {selected_scenario.description}")
    print("\n" + "-" * 50)
    print("ADVERSARY CLAIM:")
    print(f"\"{selected_scenario.adversary_claim}\"")
    print("-" * 50)
    
    # Create mock skeptical agent
    agent = MockSkepticAgent()
    
    print("\n🤖 Analyzing with skeptical agent...")
    print("⏳ Processing...")
    
    # Evaluate the scenario
    result = await benchmark.evaluate_scenario(agent, selected_scenario)
    
    print("\n📊 SKEPTICAL EVALUATION RESULTS:")
    print("=" * 50)
    print(result.response.response_text)
    
    print("\n📈 EVALUATION METRICS:")
    print(f"Overall Score: {result.metrics.overall_score:.2f}")
    print(f"Skepticism Calibration: {result.metrics.skepticism_calibration:.2f}")
    print(f"Evidence Standards: {result.metrics.evidence_standard_score:.2f}")
    print(f"Red Flag Detection: {result.metrics.red_flag_detection:.2f}")
    print(f"Reasoning Quality: {result.metrics.reasoning_quality:.2f}")
    
    print(f"\n✅ Evaluation Result: {'PASSED' if result.passed_evaluation else 'FAILED'}")
    print(f"Response Time: {result.response.response_time_ms}ms")
    
    print("\n" + "=" * 50)
    print("🎓 This demonstrates the core Agent Skeptic Bench functionality:")
    print("- Loading adversarial scenarios")  
    print("- Evaluating skeptical agent responses")
    print("- Measuring epistemic vigilance metrics")
    print("- Providing detailed feedback on reasoning quality")


def main():
    """Main entry point."""
    print("Starting Agent Skeptic Bench Demo...")
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()