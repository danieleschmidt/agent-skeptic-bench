#!/usr/bin/env python3
"""
Autonomous SDLC Demonstration Script
Terragon Autonomous SDLC Master Prompt v4.0 Implementation

This script demonstrates the complete autonomous SDLC execution cycle
with progressive enhancement and quantum optimization.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_skeptic_bench.autonomous_sdlc import AutonomousSDLC, SDLCGeneration


def setup_logging():
    """Configure logging for autonomous SDLC execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('autonomous_sdlc.log')
        ]
    )


async def demonstrate_autonomous_sdlc():
    """Demonstrate complete autonomous SDLC execution."""
    print("🚀 TERRAGON AUTONOMOUS SDLC v4.0 - DEMONSTRATION")
    print("=" * 60)
    
    # Initialize autonomous SDLC engine
    sdlc = AutonomousSDLC(project_root=Path.cwd())
    
    try:
        # Execute complete autonomous SDLC cycle
        results = await sdlc.execute_autonomous_sdlc()
        
        # Display results
        print("\n✅ AUTONOMOUS SDLC EXECUTION COMPLETE")
        print(f"Total execution time: {results['execution_time']:.2f} seconds")
        print(f"Overall success: {results['success']}")
        
        # Project Analysis Summary
        analysis = results['project_analysis']
        print(f"\n📊 PROJECT ANALYSIS:")
        print(f"  Type: {analysis.project_type.value}")
        print(f"  Language: {analysis.language}")
        print(f"  Framework: {analysis.framework}")
        print(f"  Test Coverage: {analysis.test_coverage:.1%}")
        print(f"  Security Score: {analysis.security_score:.1%}")
        
        # Generation Results
        print(f"\n🔄 GENERATION RESULTS:")
        for gen_result in results['generation_results']:
            status = "✅" if gen_result.success else "❌"
            time_taken = gen_result.end_time - gen_result.start_time if gen_result.end_time else 0
            print(f"  {status} {gen_result.generation.value.title()}: {time_taken:.2f}s")
            print(f"    Tasks completed: {len(gen_result.tasks_completed)}")
            
        # Quality Gates
        quality_results = results['quality_results']
        print(f"\n🛡️ QUALITY GATES:")
        print(f"  Passed gates: {quality_results['passed_gates']}/{quality_results['total_gates']}")
        print(f"  Critical failures: {quality_results['critical_failures']}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
        
        # Execution Summary
        summary = sdlc.get_execution_summary()
        print(f"\n📈 EXECUTION SUMMARY:")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Total Tasks: {summary['total_tasks_completed']}")
        print(f"  Status: {summary['current_status']}")
        print(f"  Quantum Enhanced: {summary['quantum_enhanced']}")
        
        return results
        
    except Exception as e:
        print(f"❌ SDLC execution failed: {e}")
        logging.error(f"SDLC execution failed: {e}", exc_info=True)
        return None


async def demonstrate_individual_generations():
    """Demonstrate individual generation execution."""
    print("\n🔬 INDIVIDUAL GENERATION DEMONSTRATION")
    print("=" * 50)
    
    sdlc = AutonomousSDLC()
    
    # Execute analysis first
    analysis = await sdlc._intelligent_analysis()
    print(f"📊 Analysis complete: {analysis.project_type.value} project")
    
    # Execute each generation individually
    for generation in SDLCGeneration:
        print(f"\n⚡ Executing {generation.value.title()}")
        result = await sdlc._execute_generation(generation)
        
        status = "✅" if result.success else "❌"
        time_taken = result.end_time - result.start_time if result.end_time else 0
        print(f"{status} Completed in {time_taken:.2f} seconds")
        print(f"   Tasks: {len(result.tasks_completed)}")
        print(f"   Quality Gates: {len([g for g in result.quality_gates if g.passed])}/{len(result.quality_gates)} passed")


async def main():
    """Main demonstration function."""
    setup_logging()
    
    print("🧠 TERRAGON LABS - AUTONOMOUS SDLC MASTER PROMPT v4.0")
    print("Intelligent Analysis + Progressive Enhancement + Autonomous Execution")
    print("=" * 80)
    
    # Run full autonomous SDLC
    results = await demonstrate_autonomous_sdlc()
    
    if results and results['success']:
        print("\n🎉 AUTONOMOUS SDLC EXECUTION SUCCESSFUL!")
        print("System is production-ready with quantum-enhanced optimization.")
    else:
        print("\n⚠️ SDLC execution completed with issues.")
        print("Review recommendations and quality gates.")
    
    # Optional: Demonstrate individual generations
    choice = input("\nDemonstrate individual generations? (y/N): ").lower()
    if choice == 'y':
        await demonstrate_individual_generations()
    
    print("\n✨ Demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())