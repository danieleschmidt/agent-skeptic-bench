"""High-level evaluation functions and utilities."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .benchmark import SkepticBenchmark
from .agents import create_skeptic_agent, BaseSkepticAgent
from .models import (
    AgentConfig,
    AgentProvider,
    BenchmarkSession,
    EvaluationResult,
    ScenarioCategory
)


logger = logging.getLogger(__name__)


class EvaluationReport:
    """Generates detailed evaluation reports."""
    
    def __init__(self, session: BenchmarkSession):
        self.session = session
        self.results = session.results
    
    def summary(self) -> str:
        """Generate text summary of evaluation results."""
        if not self.results:
            return "No evaluation results available."
        
        summary_lines = [
            f"Agent Skeptic Bench Evaluation Report",
            f"=" * 40,
            f"Session: {self.session.name}",
            f"Agent: {self.session.agent_config.model_name} ({self.session.agent_config.provider.value})",
            f"Evaluated: {self.session.total_scenarios} scenarios",
            f"Pass Rate: {self.session.pass_rate:.1%}",
            f"",
            f"Overall Metrics:",
        ]
        
        if self.session.summary_metrics:
            metrics = self.session.summary_metrics
            summary_lines.extend([
                f"  Skepticism Calibration: {metrics.skepticism_calibration:.3f}",
                f"  Evidence Standards: {metrics.evidence_standard_score:.3f}",
                f"  Red Flag Detection: {metrics.red_flag_detection:.3f}",
                f"  Reasoning Quality: {metrics.reasoning_quality:.3f}",
                f"  Overall Score: {metrics.overall_score:.3f}",
                f""
            ])
        
        # Category breakdown
        category_stats = self._get_category_stats()
        if category_stats:
            summary_lines.append("Category Breakdown:")
            for category, stats in category_stats.items():
                summary_lines.append(
                    f"  {category}: {stats['count']} scenarios, "
                    f"{stats['pass_rate']:.1%} pass rate, "
                    f"{stats['avg_score']:.3f} avg score"
                )
        
        return "\n".join(summary_lines)
    
    def detailed_analysis(self) -> Dict:
        """Generate detailed analysis of results."""
        analysis = {
            "session_info": {
                "id": self.session.id,
                "name": self.session.name,
                "agent_model": self.session.agent_config.model_name,
                "provider": self.session.agent_config.provider.value,
                "total_scenarios": self.session.total_scenarios,
                "started_at": self.session.started_at.isoformat(),
                "completed_at": self.session.completed_at.isoformat() if self.session.completed_at else None
            },
            "overall_metrics": {},
            "category_analysis": {},
            "failure_analysis": {},
            "top_performing_scenarios": [],
            "worst_performing_scenarios": []
        }
        
        if self.session.summary_metrics:
            analysis["overall_metrics"] = {
                "skepticism_calibration": self.session.summary_metrics.skepticism_calibration,
                "evidence_standard_score": self.session.summary_metrics.evidence_standard_score,
                "red_flag_detection": self.session.summary_metrics.red_flag_detection,
                "reasoning_quality": self.session.summary_metrics.reasoning_quality,
                "overall_score": self.session.summary_metrics.overall_score,
                "pass_rate": self.session.pass_rate
            }
        
        # Category analysis
        analysis["category_analysis"] = self._get_detailed_category_analysis()
        
        # Failure analysis
        analysis["failure_analysis"] = self._get_failure_analysis()
        
        # Top and worst performing scenarios
        sorted_results = sorted(self.results, key=lambda x: x.metrics.overall_score, reverse=True)
        analysis["top_performing_scenarios"] = [
            {
                "scenario_id": r.scenario.id,
                "scenario_name": r.scenario.name,
                "category": r.scenario.category.value,
                "overall_score": r.metrics.overall_score,
                "passed": r.passed_evaluation
            }
            for r in sorted_results[:5]
        ]
        analysis["worst_performing_scenarios"] = [
            {
                "scenario_id": r.scenario.id,
                "scenario_name": r.scenario.name,
                "category": r.scenario.category.value,
                "overall_score": r.metrics.overall_score,
                "passed": r.passed_evaluation
            }
            for r in sorted_results[-5:]
        ]
        
        return analysis
    
    def save_html(self, output_path: Union[str, Path]) -> None:
        """Save report as HTML file."""
        html_content = self._generate_html_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def _get_category_stats(self) -> Dict:
        """Get statistics by category."""
        category_results = {}
        for result in self.results:
            category = result.scenario.category.value
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        category_stats = {}
        for category, results in category_results.items():
            passed = sum(1 for r in results if r.passed_evaluation)
            total_score = sum(r.metrics.overall_score for r in results)
            category_stats[category] = {
                "count": len(results),
                "pass_rate": passed / len(results) if results else 0,
                "avg_score": total_score / len(results) if results else 0
            }
        
        return category_stats
    
    def _get_detailed_category_analysis(self) -> Dict:
        """Get detailed analysis by category."""
        category_analysis = {}
        category_results = {}
        
        # Group results by category
        for result in self.results:
            category = result.scenario.category.value
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Analyze each category
        for category, results in category_results.items():
            if not results:
                continue
                
            # Calculate average metrics
            avg_skepticism = sum(r.metrics.skepticism_calibration for r in results) / len(results)
            avg_evidence = sum(r.metrics.evidence_standard_score for r in results) / len(results)
            avg_red_flags = sum(r.metrics.red_flag_detection for r in results) / len(results)
            avg_reasoning = sum(r.metrics.reasoning_quality for r in results) / len(results)
            avg_overall = sum(r.metrics.overall_score for r in results) / len(results)
            
            category_analysis[category] = {
                "total_scenarios": len(results),
                "passed_scenarios": sum(1 for r in results if r.passed_evaluation),
                "pass_rate": sum(1 for r in results if r.passed_evaluation) / len(results),
                "metrics": {
                    "skepticism_calibration": avg_skepticism,
                    "evidence_standard_score": avg_evidence,
                    "red_flag_detection": avg_red_flags,
                    "reasoning_quality": avg_reasoning,
                    "overall_score": avg_overall
                },
                "common_issues": self._identify_common_issues(results)
            }
        
        return category_analysis
    
    def _get_failure_analysis(self) -> Dict:
        """Analyze common failure patterns."""
        failed_results = [r for r in self.results if not r.passed_evaluation]
        
        if not failed_results:
            return {"total_failures": 0, "failure_patterns": []}
        
        failure_patterns = []
        
        # Low skepticism calibration
        low_skepticism = [r for r in failed_results if r.metrics.skepticism_calibration < 0.3]
        if low_skepticism:
            failure_patterns.append({
                "pattern": "Low Skepticism Calibration",
                "count": len(low_skepticism),
                "description": "Agent failed to apply appropriate skepticism to implausible claims"
            })
        
        # Poor evidence standards
        poor_evidence = [r for r in failed_results if r.metrics.evidence_standard_score < 0.3]
        if poor_evidence:
            failure_patterns.append({
                "pattern": "Poor Evidence Standards",
                "count": len(poor_evidence),
                "description": "Agent failed to request appropriate quality of evidence"
            })
        
        # Missed red flags
        missed_flags = [r for r in failed_results if r.metrics.red_flag_detection < 0.3]
        if missed_flags:
            failure_patterns.append({
                "pattern": "Missed Red Flags",
                "count": len(missed_flags),
                "description": "Agent failed to identify logical fallacies and warning signs"
            })
        
        return {
            "total_failures": len(failed_results),
            "failure_rate": len(failed_results) / len(self.results),
            "failure_patterns": failure_patterns
        }
    
    def _identify_common_issues(self, results: List[EvaluationResult]) -> List[str]:
        """Identify common issues in a set of results."""
        issues = []
        
        # Check for consistently low metrics
        avg_skepticism = sum(r.metrics.skepticism_calibration for r in results) / len(results)
        avg_evidence = sum(r.metrics.evidence_standard_score for r in results) / len(results)
        avg_red_flags = sum(r.metrics.red_flag_detection for r in results) / len(results)
        
        if avg_skepticism < 0.5:
            issues.append("Consistently low skepticism calibration")
        if avg_evidence < 0.5:
            issues.append("Inadequate evidence standards")
        if avg_red_flags < 0.5:
            issues.append("Poor red flag detection")
        
        return issues
    
    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        analysis = self.detailed_analysis()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Skeptic Bench Report - {self.session.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .category-section {{ margin: 30px 0; }}
                .scenario-list {{ list-style: none; padding: 0; }}
                .scenario-item {{ background: #f9f9f9; margin: 5px 0; padding: 10px; border-radius: 3px; }}
                .pass {{ border-left: 4px solid #27ae60; }}
                .fail {{ border-left: 4px solid #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Agent Skeptic Bench Evaluation Report</h1>
                <p><strong>Session:</strong> {self.session.name}</p>
                <p><strong>Agent:</strong> {analysis['session_info']['agent_model']} ({analysis['session_info']['provider']})</p>
                <p><strong>Scenarios Evaluated:</strong> {analysis['session_info']['total_scenarios']}</p>
                <p><strong>Completed:</strong> {analysis['session_info']['completed_at'] or 'In Progress'}</p>
            </div>
        """
        
        if analysis['overall_metrics']:
            metrics = analysis['overall_metrics']
            html += f"""
            <h2>Overall Performance</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics['overall_score']:.3f}</div>
                    <div>Overall Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['pass_rate']:.1%}</div>
                    <div>Pass Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['skepticism_calibration']:.3f}</div>
                    <div>Skepticism Calibration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['evidence_standard_score']:.3f}</div>
                    <div>Evidence Standards</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['red_flag_detection']:.3f}</div>
                    <div>Red Flag Detection</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['reasoning_quality']:.3f}</div>
                    <div>Reasoning Quality</div>
                </div>
            </div>
            """
        
        # Add category analysis
        if analysis['category_analysis']:
            html += "<h2>Category Performance</h2>"
            for category, stats in analysis['category_analysis'].items():
                html += f"""
                <div class="category-section">
                    <h3>{category.replace('_', ' ').title()}</h3>
                    <p>Pass Rate: {stats['pass_rate']:.1%} ({stats['passed_scenarios']}/{stats['total_scenarios']} scenarios)</p>
                    <p>Average Score: {stats['metrics']['overall_score']:.3f}</p>
                </div>
                """
        
        html += "</body></html>"
        return html


async def run_full_evaluation(
    skeptic_agent: Union[BaseSkepticAgent, str],
    categories: Optional[List[Union[str, ScenarioCategory]]] = None,
    limit: Optional[int] = None,
    parallel: bool = True,
    concurrency: int = 5,
    save_results: Optional[Union[str, Path]] = None,
    session_name: Optional[str] = None,
    **agent_kwargs
) -> EvaluationReport:
    """Run a full evaluation of a skeptic agent.
    
    Args:
        skeptic_agent: Either a BaseSkepticAgent instance or model name string
        categories: List of categories to evaluate (default: all)
        limit: Maximum number of scenarios per category
        parallel: Whether to run evaluations in parallel
        concurrency: Number of concurrent evaluations
        save_results: Path to save results
        session_name: Name for the evaluation session
        **agent_kwargs: Additional arguments for agent creation if skeptic_agent is string
    
    Returns:
        EvaluationReport with detailed results
    """
    benchmark = SkepticBenchmark()
    
    # Handle agent creation
    if isinstance(skeptic_agent, str):
        if 'api_key' not in agent_kwargs:
            raise ValueError("api_key required when providing model name as string")
        agent = create_skeptic_agent(skeptic_agent, **agent_kwargs)
        agent_config = AgentConfig(
            provider=agent.provider,
            model_name=agent.model_name,
            api_key=agent_kwargs['api_key'],
            **{k: v for k, v in agent_kwargs.items() if k != 'api_key'}
        )
    else:
        agent = skeptic_agent
        agent_config = agent.config
    
    # Process categories
    if categories:
        processed_categories = []
        for cat in categories:
            if isinstance(cat, str):
                processed_categories.append(ScenarioCategory(cat))
            else:
                processed_categories.append(cat)
        categories = processed_categories
    else:
        categories = None
    
    # Create session
    session_name = session_name or f"Evaluation_{agent_config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = benchmark.create_session(
        name=session_name,
        agent_config=agent_config,
        description=f"Full evaluation of {agent_config.model_name}"
    )
    
    # Run evaluation
    concurrency_level = concurrency if parallel else 1
    completed_session = await benchmark.run_session(
        session,
        categories=categories,
        limit=limit,
        concurrency=concurrency_level
    )
    
    # Save results if requested
    if save_results:
        import json
        save_path = Path(save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export session data
        session_data = {
            "session_id": completed_session.id,
            "session_name": completed_session.name,
            "agent_config": {
                "provider": completed_session.agent_config.provider.value,
                "model_name": completed_session.agent_config.model_name,
                "temperature": completed_session.agent_config.temperature,
                "max_tokens": completed_session.agent_config.max_tokens
            },
            "evaluation_summary": {
                "total_scenarios": completed_session.total_scenarios,
                "passed_scenarios": completed_session.passed_scenarios,
                "pass_rate": completed_session.pass_rate,
                "started_at": completed_session.started_at.isoformat(),
                "completed_at": completed_session.completed_at.isoformat() if completed_session.completed_at else None
            },
            "overall_metrics": completed_session.summary_metrics.dict() if completed_session.summary_metrics else None,
            "results": [
                {
                    "scenario_id": r.scenario.id,
                    "scenario_name": r.scenario.name,
                    "category": r.scenario.category.value,
                    "passed": r.passed_evaluation,
                    "metrics": r.metrics.dict(),
                    "response_summary": {
                        "confidence_level": r.response.confidence_level,
                        "evidence_requests_count": len(r.response.evidence_requests),
                        "red_flags_identified_count": len(r.response.red_flags_identified),
                        "response_time_ms": r.response.response_time_ms
                    }
                }
                for r in completed_session.results
            ]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {save_path}")
    
    # Generate and return report
    report = EvaluationReport(completed_session)
    logger.info(f"Evaluation completed: {completed_session.pass_rate:.1%} pass rate")
    return report


def compare_agents(
    agent_configs: List[Dict],
    categories: Optional[List[ScenarioCategory]] = None,
    limit: Optional[int] = None,
    concurrency: int = 3
) -> Dict:
    """Compare multiple agents on the same scenarios.
    
    Args:
        agent_configs: List of agent configuration dictionaries
        categories: Categories to evaluate
        limit: Limit scenarios per category
        concurrency: Concurrent evaluations per agent
    
    Returns:
        Comparison results dictionary
    """
    async def run_comparison():
        benchmark = SkepticBenchmark()
        
        # Get scenarios once for all agents
        scenarios = benchmark.get_scenarios(categories, limit)
        
        comparison_results = {
            "scenarios_evaluated": len(scenarios),
            "categories": list(set(s.category.value for s in scenarios)),
            "agents": [],
            "scenario_results": {}
        }
        
        # Evaluate each agent
        for config in agent_configs:
            agent = create_skeptic_agent(**config)
            agent_config = AgentConfig(
                provider=agent.provider,
                model_name=agent.model_name,
                api_key=config['api_key'],
                **{k: v for k, v in config.items() if k not in ['model', 'api_key']}
            )
            
            session = benchmark.create_session(
                name=f"Comparison_{agent_config.model_name}",
                agent_config=agent_config
            )
            
            # Evaluate with fixed scenarios
            results = await benchmark.evaluate_batch(agent, scenarios, concurrency)
            
            for result in results:
                session.add_result(result)
            
            agent_summary = {
                "model": agent_config.model_name,
                "provider": agent_config.provider.value,
                "total_scenarios": len(results),
                "pass_rate": sum(1 for r in results if r.passed_evaluation) / len(results),
                "overall_score": session.summary_metrics.overall_score if session.summary_metrics else 0.0
            }
            
            if session.summary_metrics:
                agent_summary["metrics"] = {
                    "skepticism_calibration": session.summary_metrics.skepticism_calibration,
                    "evidence_standard_score": session.summary_metrics.evidence_standard_score,
                    "red_flag_detection": session.summary_metrics.red_flag_detection,
                    "reasoning_quality": session.summary_metrics.reasoning_quality
                }
            
            comparison_results["agents"].append(agent_summary)
            
            # Store per-scenario results
            for result in results:
                scenario_id = result.scenario.id
                if scenario_id not in comparison_results["scenario_results"]:
                    comparison_results["scenario_results"][scenario_id] = {
                        "scenario_name": result.scenario.name,
                        "category": result.scenario.category.value,
                        "agent_results": {}
                    }
                
                comparison_results["scenario_results"][scenario_id]["agent_results"][agent_config.model_name] = {
                    "passed": result.passed_evaluation,
                    "overall_score": result.metrics.overall_score,
                    "confidence": result.response.confidence_level
                }
        
        return comparison_results
    
    return asyncio.run(run_comparison())