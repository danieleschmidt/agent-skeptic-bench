"""Command-line interface for Agent Skeptic Bench."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .benchmark import SkepticBenchmark
from .evaluation import run_full_evaluation, compare_agents
from .models import ScenarioCategory, AgentProvider
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


def parse_categories(category_strings: List[str]) -> List[ScenarioCategory]:
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
            with open(config_file, 'r') as f:
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
        with open(args.config, 'r') as f:
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
    
    return parser


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