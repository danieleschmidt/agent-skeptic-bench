"""Analytics dashboard functionality for Agent Skeptic Bench."""

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from ..algorithms.analysis import PatternDetector, ScenarioAnalyzer, TrendAnalyzer
from ..models import EvaluationResult
from ..monitoring.metrics import get_metrics_collector
from .usage_security import UsageMetricsValidator, UsageMetricsEncryption, UsageDataRetentionManager
from .usage_monitoring import UsageMetricsMonitor

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


@dataclass
class UsageMetrics:
    """Usage metrics data structure."""
    
    timestamp: datetime
    session_id: str
    user_id: str | None = None
    agent_provider: str | None = None
    model: str | None = None
    evaluation_count: int = 0
    total_duration: float = 0.0
    api_calls: int = 0
    tokens_used: int = 0
    scenarios_completed: List[str] = None
    categories_used: List[str] = None
    performance_scores: Dict[str, float] = None
    feature_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.scenarios_completed is None:
            self.scenarios_completed = []
        if self.categories_used is None:
            self.categories_used = []
        if self.performance_scores is None:
            self.performance_scores = {}
        if self.feature_usage is None:
            self.feature_usage = {}


class UsageTracker:
    """Tracks user and system usage metrics with robust security and monitoring."""
    
    def __init__(self, storage_path: str = "data/usage_metrics", 
                 enable_encryption: bool = False, encryption_key: str | None = None):
        """Initialize usage tracker with security and monitoring."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_collector = get_metrics_collector()
        self.current_session_metrics: Dict[str, UsageMetrics] = {}
        
        # Security and validation
        self.validator = UsageMetricsValidator()
        self.encryption = UsageMetricsEncryption(encryption_key) if enable_encryption else None
        self.retention_manager = UsageDataRetentionManager()
        
        # Monitoring
        self.monitor = UsageMetricsMonitor(self.storage_path)
        
        logger.info(f"UsageTracker initialized with storage: {self.storage_path}")
    
    def start_session(self, session_id: str, user_id: str | None = None, 
                     agent_provider: str | None = None, model: str | None = None) -> None:
        """Start tracking metrics for a session with validation."""
        # Validate session parameters
        validation_errors = self.validator.validate_session_creation(session_id, user_id)
        if validation_errors:
            logger.error(f"Session validation failed: {validation_errors}")
            raise ValueError(f"Invalid session parameters: {', '.join(validation_errors)}")
        
        self.current_session_metrics[session_id] = UsageMetrics(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            user_id=user_id,
            agent_provider=agent_provider,
            model=model
        )
        
        self.metrics_collector.increment_counter("sessions_started")
        logger.info(f"Started tracking session {session_id}")
    
    def record_evaluation(self, session_id: str, scenario_id: str, category: str,
                         duration: float, score: float, tokens_used: int = 0) -> None:
        """Record evaluation metrics for a session with validation."""
        import time
        start_time = time.time()
        
        try:
            # Validate evaluation parameters
            validation_errors = self.validator.validate_evaluation_data(
                session_id, scenario_id, category, duration, score
            )
            if validation_errors:
                logger.error(f"Evaluation validation failed: {validation_errors}")
                return  # Skip recording invalid data
            
            if session_id not in self.current_session_metrics:
                logger.warning(f"Session {session_id} not found, creating new session")
                self.start_session(session_id)
            
            metrics = self.current_session_metrics[session_id]
            metrics.evaluation_count += 1
            metrics.total_duration += duration
            metrics.tokens_used += tokens_used
            metrics.scenarios_completed.append(scenario_id)
            
            if category not in metrics.categories_used:
                metrics.categories_used.append(category)
            
            if "overall_score" not in metrics.performance_scores:
                metrics.performance_scores["overall_score"] = []
            metrics.performance_scores["overall_score"].append(score)
            
            self.metrics_collector.increment_counter("evaluations_completed")
            self.metrics_collector.observe_histogram("evaluation_duration", duration)
            
        finally:
            # Record operation timing for monitoring
            operation_duration = time.time() - start_time
            self.monitor.record_operation_timing("record_evaluation", operation_duration)
        
    def record_feature_usage(self, session_id: str, feature_name: str) -> None:
        """Record usage of a specific feature."""
        if session_id not in self.current_session_metrics:
            return
            
        metrics = self.current_session_metrics[session_id]
        if feature_name not in metrics.feature_usage:
            metrics.feature_usage[feature_name] = 0
        metrics.feature_usage[feature_name] += 1
        
        self.metrics_collector.increment_counter("feature_usage", labels={"feature": feature_name})
    
    def end_session(self, session_id: str) -> UsageMetrics | None:
        """End tracking for a session and save metrics."""
        if session_id not in self.current_session_metrics:
            return None
            
        metrics = self.current_session_metrics[session_id]
        
        # Calculate aggregated performance scores
        for score_type, scores in metrics.performance_scores.items():
            if scores:
                metrics.performance_scores[f"{score_type}_avg"] = statistics.mean(scores)
                metrics.performance_scores[f"{score_type}_max"] = max(scores)
                metrics.performance_scores[f"{score_type}_min"] = min(scores)
        
        # Save to persistent storage
        self._save_metrics(metrics)
        
        # Remove from active tracking
        del self.current_session_metrics[session_id]
        
        self.metrics_collector.increment_counter("sessions_completed")
        logger.info(f"Ended tracking session {session_id}")
        
        return metrics
    
    def _save_metrics(self, metrics: UsageMetrics) -> None:
        """Save metrics to persistent storage with security and monitoring."""
        import time
        start_time = time.time()
        
        try:
            # Convert to dict for processing
            metrics_dict = asdict(metrics)
            
            # Apply encryption if enabled
            if self.encryption:
                metrics_dict = self.encryption.encrypt_sensitive_data(metrics_dict)
            
            # Sanitize data
            metrics_dict = self.validator.sanitize_user_input(metrics_dict)
            
            date_str = metrics.timestamp.strftime("%Y-%m-%d")
            file_path = self.storage_path / f"usage_metrics_{date_str}.jsonl"
            
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_dict, default=str) + "\n")
            
            # Record successful save
            self.metrics_collector.increment_counter("metrics_saved")
            logger.debug(f"Saved usage metrics for session {metrics.session_id}")
                
        except Exception as e:
            self.metrics_collector.increment_counter("metrics_save_errors")
            logger.error(f"Failed to save usage metrics: {e}")
            raise
        
        finally:
            # Record operation timing
            operation_duration = time.time() - start_time
            self.monitor.record_operation_timing("save_metrics", operation_duration)
    
    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get usage summary for the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        all_metrics = self._load_metrics_since(cutoff_date)
        
        if not all_metrics:
            return {"error": "No usage data available"}
        
        return {
            "total_sessions": len(all_metrics),
            "total_evaluations": sum(m.evaluation_count for m in all_metrics),
            "total_duration": sum(m.total_duration for m in all_metrics),
            "total_tokens": sum(m.tokens_used for m in all_metrics),
            "unique_users": len(set(m.user_id for m in all_metrics if m.user_id)),
            "popular_providers": self._get_top_providers(all_metrics),
            "popular_categories": self._get_top_categories(all_metrics),
            "avg_session_duration": statistics.mean([m.total_duration for m in all_metrics]),
            "avg_evaluations_per_session": statistics.mean([m.evaluation_count for m in all_metrics])
        }
    
    def _load_metrics_since(self, cutoff_date: datetime) -> List[UsageMetrics]:
        """Load all metrics since a specific date."""
        all_metrics = []
        
        try:
            for file_path in self.storage_path.glob("usage_metrics_*.jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            metrics_timestamp = datetime.fromisoformat(data["timestamp"])
                            
                            if metrics_timestamp >= cutoff_date:
                                # Convert back to UsageMetrics object
                                metrics = UsageMetrics(**data)
                                all_metrics.append(metrics)
                        except Exception as e:
                            logger.warning(f"Failed to parse metrics line: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to load usage metrics: {e}")
            
        return all_metrics
    
    def _get_top_providers(self, metrics: List[UsageMetrics]) -> Dict[str, int]:
        """Get top agent providers by usage."""
        provider_counts = {}
        for m in metrics:
            if m.agent_provider:
                provider_counts[m.agent_provider] = provider_counts.get(m.agent_provider, 0) + 1
        return dict(sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _get_top_categories(self, metrics: List[UsageMetrics]) -> Dict[str, int]:
        """Get top scenario categories by usage."""
        category_counts = {}
        for m in metrics:
            for category in m.categories_used:
                category_counts[category] = category_counts.get(category, 0) + 1
        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return await self.monitor.get_system_status()
    
    def cleanup_old_data(self, retention_days: int = 90) -> Dict[str, Any]:
        """Clean up old usage metrics data."""
        self.retention_manager.retention_days = retention_days
        return self.retention_manager.cleanup_old_data(self.storage_path)
    
    def archive_old_data(self, archive_path: str, retention_days: int = 90) -> Dict[str, Any]:
        """Archive old usage metrics data."""
        self.retention_manager.retention_days = retention_days
        return self.retention_manager.archive_old_data(self.storage_path, Path(archive_path))


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
