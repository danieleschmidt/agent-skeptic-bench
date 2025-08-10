"""Report generation functionality for Agent Skeptic Bench."""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..algorithms.analysis import PatternDetector, ScenarioAnalyzer, TrendAnalyzer
from ..models import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str
    description: str
    include_charts: bool = True
    include_raw_data: bool = False
    date_range: tuple[datetime, datetime] | None = None
    agent_providers: list[str] | None = None
    models: list[str] | None = None
    template: str | None = None


@dataclass
class ReportSection:
    """A section within a report."""

    title: str
    content: str
    charts: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class Report:
    """Complete generated report."""

    id: str
    title: str
    description: str
    generated_at: datetime
    config: ReportConfig
    sections: list[ReportSection]
    summary: dict[str, Any]
    metadata: dict[str, Any]


class ReportGenerator(ABC):
    """Abstract base class for report generators."""

    def __init__(self):
        """Initialize report generator."""
        self.analyzer = ScenarioAnalyzer()
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()

    @abstractmethod
    async def generate(self, results: list[EvaluationResult],
                      config: ReportConfig) -> Report:
        """Generate a report from evaluation results."""
        pass

    @abstractmethod
    async def export(self, report: Report, output_path: Path) -> None:
        """Export report to file."""
        pass

    def _generate_summary_section(self, results: list[EvaluationResult]) -> ReportSection:
        """Generate executive summary section."""
        total_evaluations = len(results)
        passed_evaluations = sum(1 for r in results if r.passed_evaluation)
        pass_rate = passed_evaluations / total_evaluations if total_evaluations > 0 else 0

        # Calculate average scores
        avg_overall = sum(r.metrics.overall_score for r in results) / total_evaluations if total_evaluations > 0 else 0
        avg_skepticism = sum(r.metrics.skepticism_calibration for r in results) / total_evaluations if total_evaluations > 0 else 0
        avg_evidence = sum(r.metrics.evidence_standard_score for r in results) / total_evaluations if total_evaluations > 0 else 0
        avg_red_flags = sum(r.metrics.red_flag_detection for r in results) / total_evaluations if total_evaluations > 0 else 0

        # Provider breakdown
        providers = {}
        for result in results:
            if result.agent_provider not in providers:
                providers[result.agent_provider] = {'count': 0, 'avg_score': 0}
            providers[result.agent_provider]['count'] += 1

        for provider in providers:
            provider_results = [r for r in results if r.agent_provider == provider]
            providers[provider]['avg_score'] = sum(r.metrics.overall_score for r in provider_results) / len(provider_results)

        content = f"""
        ## Executive Summary
        
        This report analyzes {total_evaluations} evaluations across {len(providers)} AI providers.
        
        ### Key Metrics
        - **Overall Pass Rate**: {pass_rate:.1%}
        - **Average Overall Score**: {avg_overall:.3f}
        - **Average Skepticism Calibration**: {avg_skepticism:.3f}
        - **Average Evidence Standard**: {avg_evidence:.3f}
        - **Average Red Flag Detection**: {avg_red_flags:.3f}
        
        ### Provider Performance
        """

        for provider, stats in providers.items():
            content += f"- **{provider}**: {stats['count']} evaluations, avg score {stats['avg_score']:.3f}\n"

        return ReportSection(
            title="Executive Summary",
            content=content,
            charts=[
                {
                    "type": "pie",
                    "title": "Pass Rate Distribution",
                    "data": {
                        "labels": ["Passed", "Failed"],
                        "values": [passed_evaluations, total_evaluations - passed_evaluations]
                    }
                },
                {
                    "type": "bar",
                    "title": "Average Scores by Metric",
                    "data": {
                        "labels": ["Overall", "Skepticism", "Evidence", "Red Flags"],
                        "values": [avg_overall, avg_skepticism, avg_evidence, avg_red_flags]
                    }
                }
            ],
            tables=[
                {
                    "title": "Provider Summary",
                    "headers": ["Provider", "Evaluations", "Average Score", "Pass Rate"],
                    "rows": [
                        [
                            provider,
                            stats['count'],
                            f"{stats['avg_score']:.3f}",
                            f"{sum(1 for r in results if r.agent_provider == provider and r.passed_evaluation) / stats['count']:.1%}"
                        ]
                        for provider, stats in providers.items()
                    ]
                }
            ],
            metadata={
                "total_evaluations": total_evaluations,
                "pass_rate": pass_rate,
                "providers": list(providers.keys())
            }
        )

    async def _generate_analysis_section(self, results: list[EvaluationResult]) -> ReportSection:
        """Generate detailed analysis section."""
        # Perform various analyses
        difficulty_analysis = self.analyzer.analyze_difficulty_distribution(results)
        category_analysis = self.analyzer.analyze_category_performance(results)
        patterns = self.pattern_detector.detect_response_patterns(results)
        trends = self.trend_analyzer.analyze_metric_trends(results, time_window_days=30)

        content = f"""
        ## Detailed Analysis
        
        ### Difficulty Distribution Analysis
        {chr(10).join(f"- {finding}" for finding in difficulty_analysis.findings)}
        
        ### Category Performance Analysis  
        {chr(10).join(f"- {finding}" for finding in category_analysis.findings)}
        
        ### Detected Patterns
        """

        if patterns:
            for pattern in patterns:
                content += f"- **{pattern.pattern_type}**: {pattern.description} (strength: {pattern.strength:.2f})\n"
        else:
            content += "- No significant patterns detected\n"

        content += "\n### Performance Trends\n"
        if trends:
            for trend in trends:
                content += f"- **{trend.metric_name}**: {trend.direction} trend (R² = {trend.r_squared:.3f})\n"
        else:
            content += "- No significant trends detected\n"

        return ReportSection(
            title="Detailed Analysis",
            content=content,
            charts=[
                {
                    "type": "line",
                    "title": "Performance Over Time",
                    "data": {
                        "x": [i for i in range(len(results))],
                        "y": [r.metrics.overall_score for r in results],
                        "labels": "Overall Score"
                    }
                }
            ],
            tables=[
                {
                    "title": "Analysis Summary",
                    "headers": ["Analysis Type", "Key Finding", "Confidence"],
                    "rows": [
                        ["Difficulty", difficulty_analysis.findings[0] if difficulty_analysis.findings else "No findings", f"{difficulty_analysis.confidence:.2f}"],
                        ["Category", category_analysis.findings[0] if category_analysis.findings else "No findings", f"{category_analysis.confidence:.2f}"]
                    ]
                }
            ],
            metadata={
                "difficulty_analysis": asdict(difficulty_analysis),
                "category_analysis": asdict(category_analysis),
                "patterns": [asdict(p) for p in patterns],
                "trends": [asdict(t) for t in trends]
            }
        )

    def _generate_recommendations_section(self, results: list[EvaluationResult]) -> ReportSection:
        """Generate recommendations section."""
        # Analyze results to generate recommendations
        recommendations = []

        # Check overall pass rate
        pass_rate = sum(1 for r in results if r.passed_evaluation) / len(results) if results else 0
        if pass_rate < 0.7:
            recommendations.append("Overall pass rate is below 70%. Consider reviewing evaluation criteria or providing additional training.")

        # Check for performance gaps between providers
        providers = {}
        for result in results:
            if result.agent_provider not in providers:
                providers[result.agent_provider] = []
            providers[result.agent_provider].append(result.metrics.overall_score)

        if len(providers) > 1:
            avg_scores = {p: sum(scores)/len(scores) for p, scores in providers.items()}
            max_score = max(avg_scores.values())
            min_score = min(avg_scores.values())

            if max_score - min_score > 0.2:
                best_provider = max(avg_scores, key=avg_scores.get)
                worst_provider = min(avg_scores, key=avg_scores.get)
                recommendations.append(f"Significant performance gap between {best_provider} ({avg_scores[best_provider]:.3f}) and {worst_provider} ({avg_scores[worst_provider]:.3f}). Consider investigating configuration differences.")

        # Check for low scores in specific metrics
        avg_skepticism = sum(r.metrics.skepticism_calibration for r in results) / len(results) if results else 0
        if avg_skepticism < 0.6:
            recommendations.append("Skepticism calibration scores are low. Consider scenarios that better test appropriate skepticism levels.")

        avg_red_flags = sum(r.metrics.red_flag_detection for r in results) / len(results) if results else 0
        if avg_red_flags < 0.6:
            recommendations.append("Red flag detection performance is low. Review scenarios for clear and identifiable warning signs.")

        content = "## Recommendations\n\n"
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n\n"
        else:
            content += "No specific recommendations at this time. Overall performance appears satisfactory.\n"

        return ReportSection(
            title="Recommendations",
            content=content,
            charts=[],
            tables=[
                {
                    "title": "Action Items",
                    "headers": ["Priority", "Recommendation", "Expected Impact"],
                    "rows": [
                        ["High" if "pass rate" in rec or "gap" in rec else "Medium", rec[:100] + "..." if len(rec) > 100 else rec, "Improved evaluation quality"]
                        for rec in recommendations
                    ]
                }
            ],
            metadata={
                "recommendations_count": len(recommendations),
                "pass_rate": pass_rate,
                "provider_count": len(providers)
            }
        )


class HTMLReportGenerator(ReportGenerator):
    """Generates HTML reports."""

    async def generate(self, results: list[EvaluationResult],
                      config: ReportConfig) -> Report:
        """Generate HTML report."""
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Generate sections
        sections = [
            self._generate_summary_section(results),
            await self._generate_analysis_section(results),
            self._generate_recommendations_section(results)
        ]

        # Generate overall summary
        summary = {
            "total_evaluations": len(results),
            "pass_rate": sum(1 for r in results if r.passed_evaluation) / len(results) if results else 0,
            "avg_score": sum(r.metrics.overall_score for r in results) / len(results) if results else 0,
            "date_range": {
                "start": min(r.evaluated_at for r in results) if results else None,
                "end": max(r.evaluated_at for r in results) if results else None
            }
        }

        return Report(
            id=report_id,
            title=config.title,
            description=config.description,
            generated_at=datetime.utcnow(),
            config=config,
            sections=sections,
            summary=summary,
            metadata={
                "format": "html",
                "generator_version": "1.0.0"
            }
        )

    async def export(self, report: Report, output_path: Path) -> None:
        """Export report as HTML file."""
        html_content = self._generate_html(report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report exported to {output_path}")

    def _generate_html(self, report: Report) -> str:
        """Generate HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .section {{ margin: 30px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart-placeholder {{ background: #e8f4f8; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px; }}
        .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background: #007acc; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <div class="metadata">
        <p><strong>Description:</strong> {report.description}</p>
        <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>Total Evaluations:</strong> {report.summary.get('total_evaluations', 0)}</p>
        <p><strong>Pass Rate:</strong> {report.summary.get('pass_rate', 0):.1%}</p>
        <p><strong>Average Score:</strong> {report.summary.get('avg_score', 0):.3f}</p>
    </div>
"""

        # Add sections
        for section in report.sections:
            html += f'<div class="section"><h2>{section.title}</h2>\n'
            html += f'<div>{section.content}</div>\n'

            # Add tables
            for table in section.tables:
                html += f'<h3>{table["title"]}</h3>\n<table>\n'
                html += '<tr>' + ''.join(f'<th>{header}</th>' for header in table["headers"]) + '</tr>\n'
                for row in table["rows"]:
                    html += '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>\n'
                html += '</table>\n'

            # Add chart placeholders
            for chart in section.charts:
                html += f'<div class="chart-placeholder"><strong>Chart: {chart["title"]}</strong><br>Chart data would be rendered here with a charting library</div>\n'

            html += '</div>\n'

        html += """
</body>
</html>
"""
        return html


class PDFReportGenerator(ReportGenerator):
    """Generates PDF reports."""

    async def generate(self, results: list[EvaluationResult],
                      config: ReportConfig) -> Report:
        """Generate PDF report."""
        # Use similar logic to HTML generator but prepare for PDF output
        html_generator = HTMLReportGenerator()
        report = await html_generator.generate(results, config)
        report.metadata["format"] = "pdf"
        return report

    async def export(self, report: Report, output_path: Path) -> None:
        """Export report as PDF file."""
        # In a real implementation, this would use a library like weasyprint or reportlab
        # For now, we'll create a simple text version

        content = f"""
{report.title}
{'=' * len(report.title)}

Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
Description: {report.description}

Summary:
- Total Evaluations: {report.summary.get('total_evaluations', 0)}
- Pass Rate: {report.summary.get('pass_rate', 0):.1%}
- Average Score: {report.summary.get('avg_score', 0):.3f}

"""

        for section in report.sections:
            content += f"\n{section.title}\n{'-' * len(section.title)}\n"
            content += section.content + "\n"

            for table in section.tables:
                content += f"\n{table['title']}:\n"
                # Simple table formatting
                for row in table["rows"]:
                    content += " | ".join(str(cell) for cell in row) + "\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"PDF report (as text) exported to {output_path.with_suffix('.txt')}")
        logger.warning("PDF generation requires additional dependencies (weasyprint/reportlab)")


# Factory function for creating report generators
def create_report_generator(format_type: str = "html") -> ReportGenerator:
    """Create a report generator of the specified type."""
    if format_type.lower() == "html":
        return HTMLReportGenerator()
    elif format_type.lower() == "pdf":
        return PDFReportGenerator()
    else:
        raise ValueError(f"Unsupported report format: {format_type}")
