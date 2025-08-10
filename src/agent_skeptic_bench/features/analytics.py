"""Analytics dashboard functionality for Agent Skeptic Bench."""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from ..algorithms.analysis import PatternDetector, ScenarioAnalyzer, TrendAnalyzer
from ..models import EvaluationResult

logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Time range options for analytics."""

    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of metrics for analysis."""

    OVERALL_SCORE = "overall_score"
    SKEPTICISM_CALIBRATION = "skepticism_calibration"
    EVIDENCE_STANDARD = "evidence_standard_score"
    RED_FLAG_DETECTION = "red_flag_detection"
    REASONING_QUALITY = "reasoning_quality"
    CONFIDENCE_LEVEL = "confidence_level"


@dataclass
class DashboardWidget:
    """A widget for the analytics dashboard."""

    id: str
    title: str
    widget_type: str  # "chart", "metric", "table", "text"
    data: dict[str, Any]
    config: dict[str, Any]
    last_updated: datetime


@dataclass
class DashboardConfig:
    """Configuration for analytics dashboard."""

    title: str
    time_range: TimeRange
    custom_start: datetime | None = None
    custom_end: datetime | None = None
    filters: dict[str, Any] = None
    refresh_interval: int = 300  # seconds


class AnalyticsDashboard:
    """Main analytics dashboard for Agent Skeptic Bench."""

    def __init__(self):
        """Initialize analytics dashboard."""
        self.scenario_analyzer = ScenarioAnalyzer()
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.widgets: list[DashboardWidget] = []

    async def generate_dashboard(self, results: list[EvaluationResult],
                               config: DashboardConfig) -> list[DashboardWidget]:
        """Generate complete analytics dashboard."""
        # Filter results by time range
        filtered_results = self._filter_by_time_range(results, config)

        # Generate standard widgets
        widgets = [
            await self._create_summary_metrics_widget(filtered_results),
            await self._create_performance_trends_widget(filtered_results),
            await self._create_provider_comparison_widget(filtered_results),
            await self._create_category_performance_widget(filtered_results),
            await self._create_score_distribution_widget(filtered_results),
            await self._create_recent_evaluations_widget(filtered_results),
        ]

        self.widgets = widgets
        return widgets

    def _filter_by_time_range(self, results: list[EvaluationResult],
                            config: DashboardConfig) -> list[EvaluationResult]:
        """Filter results based on time range configuration."""
        now = datetime.utcnow()

        if config.time_range == TimeRange.LAST_24_HOURS:
            cutoff = now - timedelta(hours=24)
        elif config.time_range == TimeRange.LAST_7_DAYS:
            cutoff = now - timedelta(days=7)
        elif config.time_range == TimeRange.LAST_30_DAYS:
            cutoff = now - timedelta(days=30)
        elif config.time_range == TimeRange.LAST_90_DAYS:
            cutoff = now - timedelta(days=90)
        elif config.time_range == TimeRange.CUSTOM:
            if config.custom_start and config.custom_end:
                return [r for r in results if config.custom_start <= r.evaluated_at <= config.custom_end]
            else:
                return results
        else:
            return results

        return [r for r in results if r.evaluated_at >= cutoff]

    async def _create_summary_metrics_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create summary metrics widget."""
        if not results:
            data = {"total": 0, "passed": 0, "pass_rate": 0, "avg_score": 0}
        else:
            total = len(results)
            passed = sum(1 for r in results if r.passed_evaluation)
            pass_rate = passed / total
            avg_score = sum(r.metrics.overall_score for r in results) / total

            data = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": pass_rate,
                "avg_score": avg_score,
                "metrics": {
                    "skepticism": sum(r.metrics.skepticism_calibration for r in results) / total,
                    "evidence": sum(r.metrics.evidence_standard_score for r in results) / total,
                    "red_flags": sum(r.metrics.red_flag_detection for r in results) / total,
                    "reasoning": sum(r.metrics.reasoning_quality for r in results) / total
                }
            }

        return DashboardWidget(
            id="summary_metrics",
            title="Summary Metrics",
            widget_type="metric",
            data=data,
            config={"format": "percentage_and_decimal"},
            last_updated=datetime.utcnow()
        )

    async def _create_performance_trends_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create performance trends widget."""
        if len(results) < 2:
            data = {"error": "Insufficient data for trend analysis"}
        else:
            # Sort by time
            sorted_results = sorted(results, key=lambda r: r.evaluated_at)

            # Create time series data
            timestamps = [r.evaluated_at.isoformat() for r in sorted_results]
            overall_scores = [r.metrics.overall_score for r in sorted_results]
            skepticism_scores = [r.metrics.skepticism_calibration for r in sorted_results]

            # Calculate moving averages (window of 5)
            window_size = min(5, len(sorted_results))
            overall_ma = self._calculate_moving_average(overall_scores, window_size)
            skepticism_ma = self._calculate_moving_average(skepticism_scores, window_size)

            data = {
                "timestamps": timestamps,
                "series": {
                    "overall_score": overall_scores,
                    "skepticism_calibration": skepticism_scores,
                    "overall_ma": overall_ma,
                    "skepticism_ma": skepticism_ma
                }
            }

        return DashboardWidget(
            id="performance_trends",
            title="Performance Trends",
            widget_type="chart",
            data=data,
            config={"chart_type": "line", "show_moving_average": True},
            last_updated=datetime.utcnow()
        )

    async def _create_provider_comparison_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create provider comparison widget."""
        # Group by provider
        providers = {}
        for result in results:
            if result.agent_provider not in providers:
                providers[result.agent_provider] = []
            providers[result.agent_provider].append(result)

        # Calculate statistics for each provider
        provider_stats = {}
        for provider, provider_results in providers.items():
            scores = [r.metrics.overall_score for r in provider_results]
            provider_stats[provider] = {
                "count": len(provider_results),
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0,
                "pass_rate": sum(1 for r in provider_results if r.passed_evaluation) / len(provider_results),
                "models": list(set(r.model for r in provider_results))
            }

        data = {
            "providers": list(provider_stats.keys()),
            "stats": provider_stats
        }

        return DashboardWidget(
            id="provider_comparison",
            title="Provider Comparison",
            widget_type="chart",
            data=data,
            config={"chart_type": "bar", "metric": "avg_score"},
            last_updated=datetime.utcnow()
        )

    async def _create_category_performance_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create category performance widget."""
        # Group by category
        categories = {}
        for result in results:
            category = result.scenario.category if hasattr(result, 'scenario') else 'unknown'
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Calculate performance by category
        category_performance = {}
        for category, cat_results in categories.items():
            scores = [r.metrics.overall_score for r in cat_results]
            category_performance[str(category)] = {
                "count": len(cat_results),
                "avg_score": statistics.mean(scores),
                "pass_rate": sum(1 for r in cat_results if r.passed_evaluation) / len(cat_results)
            }

        data = {
            "categories": list(category_performance.keys()),
            "performance": category_performance
        }

        return DashboardWidget(
            id="category_performance",
            title="Performance by Category",
            widget_type="chart",
            data=data,
            config={"chart_type": "radar"},
            last_updated=datetime.utcnow()
        )

    async def _create_score_distribution_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create score distribution widget."""
        if not results:
            data = {"error": "No data available"}
        else:
            scores = [r.metrics.overall_score for r in results]

            # Create histogram bins
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            histogram = [0] * (len(bins) - 1)

            for score in scores:
                for i in range(len(bins) - 1):
                    if bins[i] <= score < bins[i + 1]:
                        histogram[i] += 1
                        break
                else:  # Handle score = 1.0
                    if score == 1.0:
                        histogram[-1] += 1

            data = {
                "bins": bins,
                "histogram": histogram,
                "statistics": {
                    "mean": statistics.mean(scores),
                    "median": statistics.median(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
            }

        return DashboardWidget(
            id="score_distribution",
            title="Score Distribution",
            widget_type="chart",
            data=data,
            config={"chart_type": "histogram"},
            last_updated=datetime.utcnow()
        )

    async def _create_recent_evaluations_widget(self, results: list[EvaluationResult]) -> DashboardWidget:
        """Create recent evaluations widget."""
        # Get last 10 evaluations
        recent_results = sorted(results, key=lambda r: r.evaluated_at, reverse=True)[:10]

        table_data = []
        for result in recent_results:
            table_data.append({
                "timestamp": result.evaluated_at.strftime("%Y-%m-%d %H:%M"),
                "provider": result.agent_provider,
                "model": result.model,
                "score": f"{result.metrics.overall_score:.3f}",
                "passed": "✓" if result.passed_evaluation else "✗",
                "category": str(result.scenario.category) if hasattr(result, 'scenario') else "Unknown"
            })

        data = {
            "evaluations": table_data,
            "total_count": len(results)
        }

        return DashboardWidget(
            id="recent_evaluations",
            title="Recent Evaluations",
            widget_type="table",
            data=data,
            config={"max_rows": 10},
            last_updated=datetime.utcnow()
        )

    def _calculate_moving_average(self, values: list[float], window_size: int) -> list[float]:
        """Calculate moving average for a series of values."""
        moving_averages = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window = values[start_idx:i + 1]
            moving_averages.append(sum(window) / len(window))
        return moving_averages


class MetricsDashboard:
    """Specialized dashboard for metric analysis."""

    def __init__(self):
        """Initialize metrics dashboard."""
        pass

    async def generate_metric_analysis(self, results: list[EvaluationResult],
                                     metric_type: MetricType) -> DashboardWidget:
        """Generate analysis for a specific metric."""
        if not results:
            return DashboardWidget(
                id=f"metric_{metric_type.value}",
                title=f"{metric_type.value.replace('_', ' ').title()} Analysis",
                widget_type="text",
                data={"message": "No data available"},
                config={},
                last_updated=datetime.utcnow()
            )

        # Extract metric values
        if metric_type == MetricType.OVERALL_SCORE:
            values = [r.metrics.overall_score for r in results]
        elif metric_type == MetricType.SKEPTICISM_CALIBRATION:
            values = [r.metrics.skepticism_calibration for r in results]
        elif metric_type == MetricType.EVIDENCE_STANDARD:
            values = [r.metrics.evidence_standard_score for r in results]
        elif metric_type == MetricType.RED_FLAG_DETECTION:
            values = [r.metrics.red_flag_detection for r in results]
        elif metric_type == MetricType.REASONING_QUALITY:
            values = [r.metrics.reasoning_quality for r in results]
        elif metric_type == MetricType.CONFIDENCE_LEVEL:
            values = [r.response.confidence_level for r in results]
        else:
            values = []

        # Calculate statistics
        stats = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "q1": statistics.quantiles(values, n=4)[0] if len(values) >= 4 else values[0],
            "q3": statistics.quantiles(values, n=4)[2] if len(values) >= 4 else values[-1],
            "count": len(values)
        }

        # Time series data
        time_series = [
            {"timestamp": r.evaluated_at.isoformat(), "value": val}
            for r, val in zip(results, values, strict=False)
        ]
        time_series.sort(key=lambda x: x["timestamp"])

        data = {
            "metric_type": metric_type.value,
            "statistics": stats,
            "time_series": time_series,
            "values": values
        }

        return DashboardWidget(
            id=f"metric_{metric_type.value}",
            title=f"{metric_type.value.replace('_', ' ').title()} Analysis",
            widget_type="chart",
            data=data,
            config={"chart_type": "box_plot", "include_time_series": True},
            last_updated=datetime.utcnow()
        )


class TrendDashboard:
    """Specialized dashboard for trend analysis."""

    def __init__(self):
        """Initialize trend dashboard."""
        self.trend_analyzer = TrendAnalyzer()

    async def generate_trend_analysis(self, results: list[EvaluationResult],
                                    time_window_days: int = 30) -> list[DashboardWidget]:
        """Generate comprehensive trend analysis."""
        widgets = []

        if len(results) < 5:
            widgets.append(DashboardWidget(
                id="trend_error",
                title="Trend Analysis",
                widget_type="text",
                data={"message": "Insufficient data for trend analysis (minimum 5 evaluations required)"},
                config={},
                last_updated=datetime.utcnow()
            ))
            return widgets

        # Analyze trends for all metrics
        trends = self.trend_analyzer.analyze_metric_trends(results, time_window_days)

        # Create trend summary widget
        trend_summary = {
            "time_window": f"{time_window_days} days",
            "trends_detected": len(trends),
            "trends": [
                {
                    "metric": trend.metric_name,
                    "direction": trend.direction,
                    "significance": trend.significance,
                    "change": f"{((trend.end_value - trend.start_value) / trend.start_value * 100):.1f}%" if trend.start_value != 0 else "N/A"
                }
                for trend in trends
            ]
        }

        widgets.append(DashboardWidget(
            id="trend_summary",
            title="Trend Summary",
            widget_type="table",
            data=trend_summary,
            config={},
            last_updated=datetime.utcnow()
        ))

        # Detect anomalies
        anomalies = self.trend_analyzer.detect_anomalies(results)
        if anomalies:
            anomaly_data = {
                "count": len(anomalies),
                "anomalies": [
                    {
                        "timestamp": a.evaluated_at.isoformat(),
                        "provider": a.agent_provider,
                        "model": a.model,
                        "score": a.metrics.overall_score
                    }
                    for a in anomalies[-10:]  # Last 10 anomalies
                ]
            }

            widgets.append(DashboardWidget(
                id="anomaly_detection",
                title="Anomaly Detection",
                widget_type="table",
                data=anomaly_data,
                config={"highlight_anomalies": True},
                last_updated=datetime.utcnow()
            ))

        return widgets
