#!/usr/bin/env python3
"""
Demo of usage metrics tracking and export functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from src.agent_skeptic_bench.benchmark import SkepticBenchmark
from src.agent_skeptic_bench.features.analytics import UsageTracker
from src.agent_skeptic_bench.features.export import UsageMetricsExporter
from src.agent_skeptic_bench.models import AgentConfig, AgentProvider, ScenarioCategory


async def demo_usage_tracking():
    """Demonstrate usage metrics tracking functionality."""
    print("🎯 Demo: Usage Metrics Tracking & Export")
    print("=" * 50)
    
    # Initialize components
    benchmark = SkepticBenchmark()
    usage_tracker = UsageTracker()
    metrics_exporter = UsageMetricsExporter()
    
    print("✅ Initialized benchmark and tracking components")
    
    # Create mock agent config
    agent_config = AgentConfig(
        provider=AgentProvider.OPENAI,
        model_name="gpt-4",
        api_key="mock-key-for-demo",
        temperature=0.7
    )
    
    # Simulate user sessions
    print("\n📊 Simulating user sessions...")
    
    sessions = []
    for i in range(3):
        session_name = f"demo_session_{i+1}"
        user_id = f"user_{(i % 2) + 1}"  # Two different users
        
        # Create session with tracking
        session = benchmark.create_session(
            name=session_name,
            agent_config=agent_config,
            description=f"Demo session {i+1}",
            user_id=user_id
        )
        sessions.append(session)
        
        # Simulate feature usage
        usage_tracker.record_feature_usage(session.id, "dashboard_view")
        usage_tracker.record_feature_usage(session.id, "scenario_search")
        
        if i == 0:  # First session uses export feature
            usage_tracker.record_feature_usage(session.id, "data_export")
        
        print(f"  Created session {session.id} for {user_id}")
    
    # Simulate some evaluations
    print("\n🔍 Simulating evaluations...")
    
    for i, session in enumerate(sessions):
        # Simulate evaluations for each session
        num_evaluations = (i + 1) * 2  # 2, 4, 6 evaluations
        
        for eval_num in range(num_evaluations):
            scenario_id = f"demo_scenario_{eval_num + 1}"
            category = list(ScenarioCategory)[eval_num % len(ScenarioCategory)]
            duration = 1.5 + (eval_num * 0.3)  # Varying duration
            score = 0.7 + (eval_num * 0.05)  # Improving scores
            tokens = 150 + (eval_num * 20)  # Increasing token usage
            
            usage_tracker.record_evaluation(
                session_id=session.id,
                scenario_id=scenario_id,
                category=category.value,
                duration=duration,
                score=min(score, 1.0),  # Cap at 1.0
                tokens_used=tokens
            )
        
        print(f"  Recorded {num_evaluations} evaluations for session {session.id}")
    
    # End sessions and collect final metrics
    print("\n📈 Collecting final session metrics...")
    
    final_metrics = []
    for session in sessions:
        metrics = usage_tracker.end_session(session.id)
        if metrics:
            final_metrics.append(metrics)
            print(f"  Session {session.id}: {metrics.evaluation_count} evaluations, "
                 f"{metrics.total_duration:.1f}s duration, {metrics.tokens_used} tokens")
    
    # Generate usage summary
    print("\n📋 Usage Summary:")
    summary = usage_tracker.get_usage_summary(days=7)
    
    if "error" not in summary:
        print(f"  Total Sessions: {summary['total_sessions']}")
        print(f"  Total Evaluations: {summary['total_evaluations']}")
        print(f"  Total Duration: {summary['total_duration']:.1f}s")
        print(f"  Total Tokens: {summary['total_tokens']}")
        print(f"  Unique Users: {summary['unique_users']}")
        print(f"  Avg Session Duration: {summary['avg_session_duration']:.1f}s")
        print(f"  Popular Providers: {summary['popular_providers']}")
        print(f"  Popular Categories: {summary['popular_categories']}")
    
    # Export to different formats
    print("\n💾 Exporting usage data...")
    
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)
    
    # Export detailed usage data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON export
    json_path = output_dir / f"detailed_usage_{timestamp}.json"
    json_result = await metrics_exporter.export_detailed_usage(final_metrics, json_path, "json")
    
    if json_result.success:
        print(f"  ✅ JSON export: {json_result.file_path}")
    
    # CSV export  
    csv_path = output_dir / f"detailed_usage_{timestamp}.csv"
    csv_result = await metrics_exporter.export_detailed_usage(final_metrics, csv_path, "csv")
    
    if csv_result.success:
        print(f"  ✅ CSV export: {csv_result.file_path}")
    
    # Summary export
    summary_path = output_dir / f"usage_summary_{timestamp}.json"
    summary_result = await metrics_exporter.export_usage_summary(final_metrics, summary_path, "json")
    
    if summary_result.success:
        print(f"  ✅ Summary export: {summary_result.file_path}")
        
        # Display the summary content
        with open(summary_result.file_path, 'r') as f:
            summary_data = json.load(f)
            
        print("\n📊 Exported Summary Preview:")
        if "overall_stats" in summary_data:
            stats = summary_data["overall_stats"]
            print(f"  Total Sessions: {stats['total_sessions']}")
            print(f"  Total Evaluations: {stats['total_evaluations']}")
            print(f"  Avg Session Duration: {stats['avg_session_duration']:.1f}s")
            print(f"  Avg Evaluations/Session: {stats['avg_evaluations_per_session']:.1f}")
    
    print(f"\n🎉 Demo completed! Check the exports/ directory for generated files.")


if __name__ == "__main__":
    asyncio.run(demo_usage_tracking())