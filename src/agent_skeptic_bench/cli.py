"""Command-line interface for Agent Skeptic Bench."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .algorithms.optimization import SkepticismCalibrator
from .autonomous_sdlc import AutonomousSDLC
from .benchmark import SkepticBenchmark
from .evaluation import compare_agents, run_full_evaluation
from .models import ScenarioCategory
from .scenarios import create_default_scenario_data


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_categories(category_strings: list[str]) -> list[ScenarioCategory]:
    """Parse category strings into ScenarioCategory enum values."""
    categories = []
    for cat_str in category_strings:
        try:
            categories.append(ScenarioCategory(cat_str))
        except ValueError:
            print(f"Warning: Unknown category '{cat_str}', skipping...")
    return categories


async def cmd_evaluate(args) -> None:
    """Run evaluation command."""
    print(f"Starting evaluation with model: {args.model}")

    # Validate required arguments
    if not args.api_key:
        print("Error: API key required. Use --api-key or set environment variable.")
        sys.exit(1)

    # Parse categories
    categories = None
    if args.categories:
        categories = parse_categories(args.categories)
        if not categories:
            print("Error: No valid categories specified.")
            sys.exit(1)

    # Determine provider
    provider = args.provider
    if not provider:
        if 'gpt' in args.model.lower():
            provider = 'openai'
        elif 'claude' in args.model.lower():
            provider = 'anthropic'
        elif 'gemini' in args.model.lower():
            provider = 'google'
        else:
            print("Error: Cannot auto-detect provider. Please specify --provider.")
            sys.exit(1)

    try:
        # Run evaluation
        report = await run_full_evaluation(
            skeptic_agent=args.model,
            api_key=args.api_key,
            provider=provider,
            categories=categories,
            limit=args.limit,
            parallel=not args.no_parallel,
            concurrency=args.concurrency,
            save_results=args.output,
            session_name=args.session_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

        # Print summary
        print("\n" + "="*50)
        print(report.summary())
        print("="*50)

        # Save HTML report if requested
        if args.html_report:
            report.save_html(args.html_report)
            print(f"HTML report saved to: {args.html_report}")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def cmd_compare(args) -> None:
    """Run comparison command."""
    print("Starting model comparison...")

    # Load configurations from files
    agent_configs = []
    for config_file in args.configs:
        try:
            with open(config_file) as f:
                config = json.load(f)
                agent_configs.append(config)
        except Exception as e:
            print(f"Error loading config {config_file}: {e}")
            sys.exit(1)

    if len(agent_configs) < 2:
        print("Error: At least 2 agent configurations required for comparison.")
        sys.exit(1)

    # Parse categories
    categories = None
    if args.categories:
        categories = parse_categories(args.categories)

    try:
        # Run comparison
        results = await compare_agents(
            agent_configs=agent_configs,
            categories=categories,
            limit=args.limit,
            concurrency=args.concurrency
        )

        # Print results
        print("\n" + "="*50)
        print("AGENT COMPARISON RESULTS")
        print("="*50)
        print(f"Scenarios evaluated: {results['scenarios_evaluated']}")
        print(f"Categories: {', '.join(results['categories'])}")
        print()

        # Sort agents by overall score
        agents = sorted(results['agents'], key=lambda x: x['overall_score'], reverse=True)

        print("LEADERBOARD:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Model':<20} {'Provider':<10} {'Score':<8} {'Pass Rate':<10}")
        print("-" * 70)

        for i, agent in enumerate(agents, 1):
            print(f"{i:<4} {agent['model']:<20} {agent['provider']:<10} "
                  f"{agent['overall_score']:<8.3f} {agent['pass_rate']:<10.1%}")

        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")

    except Exception as e:
        print(f"Comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_list_scenarios(args) -> None:
    """List available scenarios."""
    benchmark = SkepticBenchmark()

    # Get scenarios
    categories = None
    if args.categories:
        categories = parse_categories(args.categories)

    scenarios = benchmark.get_scenarios(categories, args.limit)

    print(f"Found {len(scenarios)} scenarios")
    print("-" * 50)

    # Group by category
    by_category = {}
    for scenario in scenarios:
        cat = scenario.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(scenario)

    for category, cat_scenarios in by_category.items():
        print(f"\n{category.upper()} ({len(cat_scenarios)} scenarios):")
        for scenario in cat_scenarios:
            difficulty = scenario.metadata.get('difficulty', 'unknown')
            print(f"  {scenario.id}: {scenario.name} [{difficulty}]")
            if args.verbose:
                print(f"    Description: {scenario.description}")
                print(f"    Skepticism Level: {scenario.correct_skepticism_level}")


def cmd_generate_data(args) -> None:
    """Generate default scenario data files."""
    output_path = Path(args.output)
    print(f"Generating scenario data files in: {output_path}")

    try:
        create_default_scenario_data(output_path)
        print("Scenario data files generated successfully!")
    except Exception as e:
        print(f"Failed to generate data: {e}")
        sys.exit(1)


def cmd_validate_config(args) -> None:
    """Validate agent configuration file."""
    try:
        with open(args.config) as f:
            config = json.load(f)

        required_fields = ['model', 'api_key']
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            print(f"Invalid configuration: Missing fields {missing_fields}")
            sys.exit(1)

        # Try to create agent
        from .agents import create_skeptic_agent
        agent = create_skeptic_agent(**config)

        print("Configuration is valid!")
        print(f"Provider: {agent.provider.value}")
        print(f"Model: {agent.model_name}")

    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)


async def cmd_optimize(args) -> None:
    """Run quantum optimization command."""
    print("Starting quantum-inspired skepticism optimization...")

    if not args.evaluation_data:
        print("Error: Evaluation data file required for optimization.")
        sys.exit(1)

    # Load evaluation data
    try:
        with open(args.evaluation_data) as f:
            evaluation_data = json.load(f)
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        sys.exit(1)

    # Initialize quantum optimizer
    calibrator = SkepticismCalibrator()

    # Mock evaluation data structure for demonstration
    # In practice, this would be loaded from actual evaluation results
    mock_evaluations = []

    try:
        # Run quantum optimization
        print("Initializing quantum population...")
        optimal_params = calibrator.calibrate_agent_parameters(
            historical_evaluations=mock_evaluations,
            target_metrics=args.target_metrics
        )

        # Generate report
        report = calibrator.get_calibration_report()

        print("\n" + "="*50)
        print("QUANTUM OPTIMIZATION RESULTS")
        print("="*50)
        print("Optimal Parameters:")
        for param, value in optimal_params.items():
            print(f"  {param}: {value:.4f}")

        print("\nOptimization Performance:")
        perf = report.get("optimization_performance", {})
        print(f"  Final Fitness: {perf.get('average_final_fitness', 0):.4f}")
        print(f"  Stability: {perf.get('optimization_stability', 0):.4f}")

        print("\nRecommendations:")
        for rec in report.get("recommendations", []):
            print(f"  • {rec}")

        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "optimal_parameters": optimal_params,
                    "calibration_report": report
                }, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Optimization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def cmd_predict_skepticism(args) -> None:
    """Predict optimal skepticism for a scenario."""
    if not args.scenario_file:
        print("Error: Scenario file required.")
        sys.exit(1)

    if not args.parameters:
        print("Error: Agent parameters file required.")
        sys.exit(1)

    try:
        # Load scenario
        with open(args.scenario_file) as f:
            scenario_data = json.load(f)

        # Load parameters
        with open(args.parameters) as f:
            parameters = json.load(f)

        # Initialize calibrator
        calibrator = SkepticismCalibrator()

        # Mock scenario object for demonstration
        from .models import Scenario, ScenarioCategory
        scenario = Scenario(
            id=scenario_data.get("id", "test"),
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title=scenario_data.get("title", "Test Scenario"),
            description=scenario_data.get("description", ""),
            correct_skepticism_level=scenario_data.get("correct_skepticism_level", 0.5),
            metadata=scenario_data.get("metadata", {})
        )

        # Predict optimal skepticism
        predicted_skepticism = calibrator.predict_optimal_skepticism(scenario, parameters)

        print(f"Scenario: {scenario.title}")
        print(f"Predicted Optimal Skepticism: {predicted_skepticism:.4f}")
        print(f"Target Skepticism: {scenario.correct_skepticism_level:.4f}")

        accuracy = 1.0 - abs(predicted_skepticism - scenario.correct_skepticism_level)
        print(f"Prediction Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Agent Skeptic Bench - Evaluate AI agents' epistemic vigilance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate GPT-4 on all scenarios
  agent-skeptic-bench evaluate --model gpt-4 --api-key $OPENAI_API_KEY

  # Evaluate Claude on specific categories with output
  agent-skeptic-bench evaluate --model claude-3-opus \\
    --api-key $ANTHROPIC_API_KEY \\
    --categories factual_claims flawed_plans \\
    --output results/claude_results.json

  # Compare multiple models
  agent-skeptic-bench compare \\
    --configs config/gpt4.json config/claude.json \\
    --output comparison_results.json

  # List available scenarios
  agent-skeptic-bench list-scenarios --categories factual_claims
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single agent')
    eval_parser.add_argument('--model', required=True,
                            help='Model name (e.g., gpt-4, claude-3-opus)')
    eval_parser.add_argument('--api-key',
                            help='API key (can also use environment variable)')
    eval_parser.add_argument('--provider',
                            choices=['openai', 'anthropic', 'google'],
                            help='Provider (auto-detected if not specified)')
    eval_parser.add_argument('--categories', nargs='+',
                            choices=[cat.value for cat in ScenarioCategory],
                            help='Categories to evaluate')
    eval_parser.add_argument('--limit', type=int,
                            help='Limit number of scenarios per category')
    eval_parser.add_argument('--no-parallel', action='store_true',
                            help='Disable parallel evaluation')
    eval_parser.add_argument('--concurrency', type=int, default=5,
                            help='Number of concurrent evaluations (default: 5)')
    eval_parser.add_argument('--temperature', type=float, default=0.7,
                            help='Model temperature (default: 0.7)')
    eval_parser.add_argument('--max-tokens', type=int, default=1000,
                            help='Maximum tokens per response (default: 1000)')
    eval_parser.add_argument('--output', '-o',
                            help='Output file for results (JSON)')
    eval_parser.add_argument('--html-report',
                            help='Output file for HTML report')
    eval_parser.add_argument('--session-name',
                            help='Name for evaluation session')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple agents')
    compare_parser.add_argument('--configs', nargs='+', required=True,
                               help='Agent configuration files (JSON)')
    compare_parser.add_argument('--categories', nargs='+',
                               choices=[cat.value for cat in ScenarioCategory],
                               help='Categories to evaluate')
    compare_parser.add_argument('--limit', type=int,
                               help='Limit number of scenarios per category')
    compare_parser.add_argument('--concurrency', type=int, default=3,
                               help='Concurrent evaluations per agent (default: 3)')
    compare_parser.add_argument('--output', '-o',
                               help='Output file for comparison results (JSON)')

    # List scenarios command
    list_parser = subparsers.add_parser('list-scenarios', help='List available scenarios')
    list_parser.add_argument('--categories', nargs='+',
                            choices=[cat.value for cat in ScenarioCategory],
                            help='Categories to list')
    list_parser.add_argument('--limit', type=int,
                            help='Limit number of scenarios to show')

    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate default scenario data')
    gen_parser.add_argument('--output', '-o', default='./data/scenarios',
                           help='Output directory for scenario data (default: ./data/scenarios)')

    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate agent configuration')
    validate_parser.add_argument('--config', required=True,
                                help='Configuration file to validate')

    # Quantum optimization command
    optimize_parser = subparsers.add_parser('quantum-optimize', help='Run quantum-inspired optimization')
    optimize_parser.add_argument('--evaluation-data', required=True,
                                help='Evaluation data file (JSON)')
    optimize_parser.add_argument('--target-metrics', type=dict,
                                help='Target metrics for optimization')
    optimize_parser.add_argument('--output', '-o',
                                help='Output file for optimization results')

    # Skepticism prediction command
    predict_parser = subparsers.add_parser('predict-skepticism', help='Predict optimal skepticism level')
    predict_parser.add_argument('--scenario-file', required=True,
                               help='Scenario data file (JSON)')
    predict_parser.add_argument('--parameters', required=True,
                               help='Agent parameters file (JSON)')

    # Autonomous SDLC commands
    sdlc_parser = subparsers.add_parser('autonomous-sdlc', help='Execute autonomous SDLC cycle')
    sdlc_parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                            help='Project root directory')
    sdlc_parser.add_argument('--generation', choices=['1', '2', '3', 'all'], default='all',
                            help='Execute specific generation or all')
    sdlc_parser.add_argument('--output', '-o',
                            help='Output file for SDLC results (JSON)')
    
    # SDLC status command
    status_parser = subparsers.add_parser('sdlc-status', help='Check autonomous SDLC status')
    status_parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                              help='Project root directory')

    # Export usage metrics command
    export_parser = subparsers.add_parser('export-usage', help='Export usage metrics and analytics')
    export_parser.add_argument('--days', type=int, default=7,
                              help='Number of days of data to export (default: 7)')
    export_parser.add_argument('--format', choices=['json', 'csv', 'excel'], default='json',
                              help='Export format (default: json)')
    export_parser.add_argument('--output', '-o', 
                              help='Output file path (auto-generated if not specified)')
    export_parser.add_argument('--summary-only', action='store_true',
                              help='Export only summary analytics, not detailed data')
    export_parser.add_argument('--include-trends', action='store_true',
                              help='Include performance trend analysis')

    return parser


async def cmd_autonomous_sdlc(args) -> None:
    """Execute autonomous SDLC cycle."""
    print("🚀 Starting Autonomous SDLC Execution")
    print(f"Project root: {args.project_root}")
    print(f"Generation: {args.generation}")
    
    try:
        # Initialize autonomous SDLC
        sdlc = AutonomousSDLC(project_root=args.project_root)
        
        if args.generation == 'all':
            # Execute full autonomous SDLC cycle
            results = await sdlc.execute_autonomous_sdlc()
            
            print(f"\n✅ Autonomous SDLC completed in {results['execution_time']:.2f}s")
            print(f"Overall success: {results['success']}")
            
            # Display generation results
            for gen_result in results['generation_results']:
                status = "✅" if gen_result.success else "❌"
                time_taken = gen_result.end_time - gen_result.start_time if gen_result.end_time else 0
                print(f"{status} {gen_result.generation.value.title()}: {time_taken:.2f}s")
        else:
            # Execute specific generation
            from .autonomous_sdlc import SDLCGeneration
            generation_map = {
                '1': SDLCGeneration.GENERATION_1_WORK,
                '2': SDLCGeneration.GENERATION_2_ROBUST,
                '3': SDLCGeneration.GENERATION_3_SCALE
            }
            
            generation = generation_map[args.generation]
            result = await sdlc._execute_generation(generation)
            
            status = "✅" if result.success else "❌"
            time_taken = result.end_time - result.start_time if result.end_time else 0
            print(f"{status} {generation.value.title()} completed in {time_taken:.2f}s")
            print(f"Tasks completed: {len(result.tasks_completed)}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results if args.generation == 'all' else result.__dict__, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"❌ SDLC execution failed: {e}")
        sys.exit(1)


def cmd_sdlc_status(args) -> None:
    """Check autonomous SDLC status."""
    print("📊 Autonomous SDLC Status")
    print(f"Project root: {args.project_root}")
    
    try:
        # Initialize and get status
        sdlc = AutonomousSDLC(project_root=args.project_root)
        summary = sdlc.get_execution_summary()
        
        if summary['status'] == 'not_executed':
            print("❌ No SDLC execution history found")
            print("Run 'autonomous-sdlc' command to execute SDLC cycle")
            return
        
        print(f"Status: {summary['current_status']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Successful Generations: {summary['successful_generations']}")
        print(f"Total Tasks Completed: {summary['total_tasks_completed']}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"Quantum Enhanced: {summary['quantum_enhanced']}")
        
        if summary['project_analysis']:
            analysis = summary['project_analysis']
            print(f"\nProject Analysis:")
            print(f"  Type: {analysis.project_type.value}")
            print(f"  Language: {analysis.language}")
            print(f"  Test Coverage: {analysis.test_coverage:.1%}")
            
    except Exception as e:
        print(f"❌ Failed to get SDLC status: {e}")
        sys.exit(1)


async def cmd_export_usage(args) -> None:
    """Export usage metrics and analytics."""
    from .features.analytics import UsageTracker
    from .features.export import UsageMetricsExporter, AdvancedAnalyticsExporter
    from datetime import datetime, timedelta
    
    print(f"📊 Exporting usage metrics for last {args.days} days")
    
    try:
        # Initialize components
        usage_tracker = UsageTracker()
        metrics_exporter = UsageMetricsExporter()
        analytics_exporter = AdvancedAnalyticsExporter()
        
        # Load usage data
        cutoff_date = datetime.utcnow() - timedelta(days=args.days)
        usage_data = usage_tracker._load_metrics_since(cutoff_date)
        
        if not usage_data:
            print("❌ No usage data found for the specified time period")
            return
        
        print(f"Found {len(usage_data)} usage records")
        
        # Generate output path if not specified
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = "xlsx" if args.format == "excel" else args.format
            args.output = f"usage_metrics_{timestamp}.{file_extension}"
        
        output_path = Path(args.output)
        
        # Export based on mode
        if args.summary_only:
            # Export summary analytics
            result = await metrics_exporter.export_usage_summary(
                usage_data, output_path, args.format
            )
        else:
            # Export detailed usage data
            result = await metrics_exporter.export_detailed_usage(
                usage_data, output_path, args.format
            )
        
        if result.success:
            print(f"✅ Successfully exported {result.records_exported} records to {result.file_path}")
            
            # Export trends if requested
            if args.include_trends:
                trends_path = output_path.with_stem(f"{output_path.stem}_trends").with_suffix('.json')
                
                # We need evaluation results for trends - this would need to be loaded from a results database
                print("⚠️  Trend analysis requires evaluation results data (not yet implemented)")
                
        else:
            print(f"❌ Export failed: {result.error_message}")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Failed to export usage metrics: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    try:
        if args.command == 'evaluate':
            asyncio.run(cmd_evaluate(args))
        elif args.command == 'compare':
            asyncio.run(cmd_compare(args))
        elif args.command == 'list-scenarios':
            cmd_list_scenarios(args)
        elif args.command == 'generate-data':
            cmd_generate_data(args)
        elif args.command == 'validate-config':
            cmd_validate_config(args)
        elif args.command == 'quantum-optimize':
            asyncio.run(cmd_optimize(args))
        elif args.command == 'predict-skepticism':
            cmd_predict_skepticism(args)
        elif args.command == 'autonomous-sdlc':
            asyncio.run(cmd_autonomous_sdlc(args))
        elif args.command == 'export-usage':
            asyncio.run(cmd_export_usage(args))
        elif args.command == 'sdlc-status':
            cmd_sdlc_status(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
