"""Advanced analysis algorithms for evaluation data."""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..models import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Results from analysis algorithms."""

    name: str
    findings: list[str]
    metrics: dict[str, float]
    recommendations: list[str]
    confidence: float
    data: dict[str, Any] = None


@dataclass
class Pattern:
    """Represents a detected pattern in evaluation data."""

    pattern_type: str
    description: str
    strength: float  # 0.0 to 1.0
    occurrences: int
    examples: list[str]
    metadata: dict[str, Any] = None


@dataclass
class Trend:
    """Represents a trend in evaluation metrics over time."""

    metric_name: str
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    r_squared: float
    start_value: float
    end_value: float
    time_period: str
    significance: float


class ScenarioAnalyzer:
    """Analyzes scenario performance and characteristics."""

    def __init__(self):
        """Initialize scenario analyzer."""
        self.scaler = StandardScaler()

    def analyze_difficulty_distribution(self, results: list[EvaluationResult]) -> AnalysisResult:
        """Analyze distribution of scenario difficulties."""
        if not results:
            return AnalysisResult(
                name="Difficulty Distribution",
                findings=["No evaluation results available"],
                metrics={},
                recommendations=[],
                confidence=0.0
            )

        # Extract difficulty levels and scores
        difficulty_scores = defaultdict(list)
        for result in results:
            difficulty = result.scenario.metadata.get('difficulty', 'unknown')
            difficulty_scores[difficulty].append(result.metrics.overall_score)

        findings = []
        metrics = {}
        recommendations = []

        # Calculate statistics for each difficulty level
        for difficulty, scores in difficulty_scores.items():
            if scores:
                mean_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

                metrics[f"{difficulty}_mean_score"] = mean_score
                metrics[f"{difficulty}_std_score"] = std_score
                metrics[f"{difficulty}_count"] = len(scores)

                findings.append(
                    f"{difficulty.title()} scenarios: {len(scores)} total, "
                    f"mean score {mean_score:.3f} ± {std_score:.3f}"
                )

        # Analyze difficulty progression
        if len(difficulty_scores) > 1:
            ordered_difficulties = ['easy', 'medium', 'hard']
            available_difficulties = [d for d in ordered_difficulties if d in difficulty_scores]

            if len(available_difficulties) >= 2:
                scores_by_order = [
                    statistics.mean(difficulty_scores[d])
                    for d in available_difficulties
                ]

                # Check if scores decrease with difficulty (expected pattern)
                is_decreasing = all(
                    scores_by_order[i] >= scores_by_order[i+1]
                    for i in range(len(scores_by_order)-1)
                )

                if is_decreasing:
                    findings.append("Scenario difficulty progression is well-calibrated")
                else:
                    findings.append("Scenario difficulty progression may need recalibration")
                    recommendations.append("Review difficulty assignments for scenarios")

        # Check for outliers
        all_scores = [score for scores in difficulty_scores.values() for score in scores]
        if all_scores:
            q1 = np.percentile(all_scores, 25)
            q3 = np.percentile(all_scores, 75)
            iqr = q3 - q1
            outliers = [s for s in all_scores if s < q1 - 1.5*iqr or s > q3 + 1.5*iqr]

            if outliers:
                findings.append(f"Found {len(outliers)} outlier scores")
                if len(outliers) > len(all_scores) * 0.1:
                    recommendations.append("Investigate scenarios with outlier scores")

        confidence = min(1.0, len(results) / 100)  # Higher confidence with more data

        return AnalysisResult(
            name="Difficulty Distribution",
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data={
                "difficulty_scores": dict(difficulty_scores),
                "outliers": outliers if 'outliers' in locals() else []
            }
        )

    def analyze_category_performance(self, results: list[EvaluationResult]) -> AnalysisResult:
        """Analyze performance across scenario categories."""
        if not results:
            return AnalysisResult(
                name="Category Performance",
                findings=["No evaluation results available"],
                metrics={},
                recommendations=[],
                confidence=0.0
            )

        # Group results by category
        category_results = defaultdict(list)
        for result in results:
            category_results[result.scenario.category].append(result)

        findings = []
        metrics = {}
        recommendations = []

        # Calculate performance metrics for each category
        category_stats = {}
        for category, cat_results in category_results.items():
            scores = [r.metrics.overall_score for r in cat_results]
            pass_rates = [r.passed_evaluation for r in cat_results]

            stats_dict = {
                'count': len(cat_results),
                'mean_score': statistics.mean(scores),
                'median_score': statistics.median(scores),
                'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'pass_rate': sum(pass_rates) / len(pass_rates),
                'min_score': min(scores),
                'max_score': max(scores)
            }

            category_stats[category.value] = stats_dict

            # Update metrics dictionary
            for key, value in stats_dict.items():
                metrics[f"{category.value}_{key}"] = value

            findings.append(
                f"{category.value}: {stats_dict['count']} scenarios, "
                f"mean score {stats_dict['mean_score']:.3f}, "
                f"pass rate {stats_dict['pass_rate']:.1%}"
            )

        # Identify strongest and weakest categories
        if len(category_stats) > 1:
            sorted_categories = sorted(
                category_stats.items(),
                key=lambda x: x[1]['mean_score'],
                reverse=True
            )

            strongest = sorted_categories[0]
            weakest = sorted_categories[-1]

            findings.append(f"Strongest category: {strongest[0]} (score: {strongest[1]['mean_score']:.3f})")
            findings.append(f"Weakest category: {weakest[0]} (score: {weakest[1]['mean_score']:.3f})")

            # Check for significant differences
            score_diff = strongest[1]['mean_score'] - weakest[1]['mean_score']
            if score_diff > 0.2:
                recommendations.append(
                    f"Large performance gap between {strongest[0]} and {weakest[0]} categories"
                )
                recommendations.append(f"Consider additional training on {weakest[0]} scenarios")

        # Statistical tests for category differences
        if len(category_stats) > 2:
            try:
                # Perform ANOVA to test for significant differences
                category_scores = [
                    [r.metrics.overall_score for r in cat_results]
                    for cat_results in category_results.values()
                ]

                f_stat, p_value = stats.f_oneway(*category_scores)
                metrics['anova_f_statistic'] = f_stat
                metrics['anova_p_value'] = p_value

                if p_value < 0.05:
                    findings.append("Statistically significant differences found between categories")
                else:
                    findings.append("No statistically significant differences between categories")

            except Exception as e:
                logger.warning(f"Failed to perform ANOVA: {e}")

        confidence = min(1.0, len(results) / 50)

        return AnalysisResult(
            name="Category Performance",
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data={"category_stats": category_stats}
        )

    def detect_scenario_clusters(self, results: list[EvaluationResult], n_clusters: int = 3) -> AnalysisResult:
        """Detect clusters of similar scenarios based on performance patterns."""
        if len(results) < n_clusters * 2:
            return AnalysisResult(
                name="Scenario Clustering",
                findings=["Insufficient data for clustering analysis"],
                metrics={},
                recommendations=[],
                confidence=0.0
            )

        # Extract features for clustering
        features = []
        scenario_ids = []

        for result in results:
            feature_vector = [
                result.metrics.skepticism_calibration,
                result.metrics.evidence_standard_score,
                result.metrics.red_flag_detection,
                result.metrics.reasoning_quality,
                result.response.confidence_level,
                len(result.response.evidence_requests),
                len(result.response.red_flags_identified),
                result.scenario.correct_skepticism_level
            ]
            features.append(feature_vector)
            scenario_ids.append(result.scenario.id)

        features_array = np.array(features)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features_array)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Analyze clusters
        findings = []
        metrics = {}
        recommendations = []

        cluster_analysis = defaultdict(list)
        for i, (result, label) in enumerate(zip(results, cluster_labels, strict=False)):
            cluster_analysis[label].append(result)

        # Characterize each cluster
        for cluster_id, cluster_results in cluster_analysis.items():
            cluster_size = len(cluster_results)
            avg_score = statistics.mean([r.metrics.overall_score for r in cluster_results])

            metrics[f"cluster_{cluster_id}_size"] = cluster_size
            metrics[f"cluster_{cluster_id}_avg_score"] = avg_score

            # Find dominant characteristics
            categories = [r.scenario.category.value for r in cluster_results]
            dominant_category = max(set(categories), key=categories.count)

            findings.append(
                f"Cluster {cluster_id}: {cluster_size} scenarios, "
                f"avg score {avg_score:.3f}, "
                f"dominant category: {dominant_category}"
            )

        # Calculate silhouette score for clustering quality
        try:
            from sklearn.metrics import silhouette_score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            metrics['silhouette_score'] = silhouette_avg

            if silhouette_avg > 0.5:
                findings.append("Good cluster separation achieved")
            else:
                findings.append("Clusters may be overlapping")

        except ImportError:
            logger.warning("sklearn.metrics not available for silhouette score")

        confidence = min(1.0, len(results) / 100)

        return AnalysisResult(
            name="Scenario Clustering",
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence,
            data={
                "cluster_labels": cluster_labels.tolist(),
                "scenario_ids": scenario_ids,
                "cluster_centers": kmeans.cluster_centers_.tolist()
            }
        )


class PatternDetector:
    """Detects patterns in evaluation responses and behaviors."""

    def __init__(self):
        """Initialize pattern detector."""
        pass

    def detect_response_patterns(self, results: list[EvaluationResult]) -> list[Pattern]:
        """Detect patterns in agent responses."""
        patterns = []

        if not results:
            return patterns

        # Pattern 1: Consistent over/under confidence
        confidence_levels = [r.response.confidence_level for r in results]
        if confidence_levels:
            mean_confidence = statistics.mean(confidence_levels)
            std_confidence = statistics.stdev(confidence_levels) if len(confidence_levels) > 1 else 0.0

            if mean_confidence > 0.8 and std_confidence < 0.1:
                patterns.append(Pattern(
                    pattern_type="overconfidence",
                    description="Agent consistently shows high confidence",
                    strength=min(1.0, (mean_confidence - 0.8) / 0.2),
                    occurrences=len([c for c in confidence_levels if c > 0.8]),
                    examples=[f"Confidence: {c:.2f}" for c in confidence_levels[:3]]
                ))
            elif mean_confidence < 0.3 and std_confidence < 0.1:
                patterns.append(Pattern(
                    pattern_type="underconfidence",
                    description="Agent consistently shows low confidence",
                    strength=min(1.0, (0.3 - mean_confidence) / 0.3),
                    occurrences=len([c for c in confidence_levels if c < 0.3]),
                    examples=[f"Confidence: {c:.2f}" for c in confidence_levels[:3]]
                ))

        # Pattern 2: Evidence request patterns
        evidence_counts = [len(r.response.evidence_requests) for r in results]
        if evidence_counts:
            avg_evidence_requests = statistics.mean(evidence_counts)

            if avg_evidence_requests < 0.5:
                patterns.append(Pattern(
                    pattern_type="minimal_evidence_requests",
                    description="Agent rarely requests evidence",
                    strength=1.0 - avg_evidence_requests,
                    occurrences=len([c for c in evidence_counts if c == 0]),
                    examples=["No evidence requested"] * min(3, len(evidence_counts))
                ))
            elif avg_evidence_requests > 3.0:
                patterns.append(Pattern(
                    pattern_type="excessive_evidence_requests",
                    description="Agent requests excessive evidence",
                    strength=min(1.0, (avg_evidence_requests - 3.0) / 5.0),
                    occurrences=len([c for c in evidence_counts if c > 3]),
                    examples=[f"{c} evidence requests" for c in evidence_counts[:3]]
                ))

        # Pattern 3: Red flag detection patterns
        red_flag_counts = [len(r.response.red_flags_identified) for r in results]
        correct_red_flags = [len(r.scenario.red_flags) for r in results]

        if red_flag_counts and correct_red_flags:
            detection_rates = [
                min(1.0, detected / max(1, expected))
                for detected, expected in zip(red_flag_counts, correct_red_flags, strict=False)
            ]
            avg_detection_rate = statistics.mean(detection_rates)

            if avg_detection_rate < 0.3:
                patterns.append(Pattern(
                    pattern_type="poor_red_flag_detection",
                    description="Agent misses many red flags",
                    strength=1.0 - avg_detection_rate,
                    occurrences=len([r for r in detection_rates if r < 0.3]),
                    examples=[f"Detection rate: {r:.1%}" for r in detection_rates[:3]]
                ))

        # Pattern 4: Category-specific performance patterns
        category_performance = defaultdict(list)
        for result in results:
            category_performance[result.scenario.category].append(result.metrics.overall_score)

        for category, scores in category_performance.items():
            if len(scores) >= 3:
                avg_score = statistics.mean(scores)
                if avg_score < 0.4:
                    patterns.append(Pattern(
                        pattern_type="category_weakness",
                        description=f"Consistent poor performance in {category.value}",
                        strength=1.0 - avg_score,
                        occurrences=len(scores),
                        examples=[f"{category.value}: {s:.2f}" for s in scores[:3]],
                        metadata={"category": category.value}
                    ))

        return patterns

    def detect_temporal_patterns(self, results: list[EvaluationResult]) -> list[Pattern]:
        """Detect temporal patterns in evaluation performance."""
        patterns = []

        if len(results) < 10:
            return patterns

        # Sort results by evaluation time
        sorted_results = sorted(results, key=lambda r: r.evaluated_at)

        # Pattern 1: Learning/deterioration over time
        scores = [r.metrics.overall_score for r in sorted_results]
        time_indices = list(range(len(scores)))

        if len(scores) > 5:
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, scores)

            if abs(r_value) > 0.3 and p_value < 0.05:
                if slope > 0.01:
                    patterns.append(Pattern(
                        pattern_type="performance_improvement",
                        description="Performance improves over time",
                        strength=min(1.0, abs(r_value)),
                        occurrences=len(scores),
                        examples=[f"Score trend: {slope:.3f} per evaluation"],
                        metadata={"slope": slope, "r_squared": r_value**2}
                    ))
                elif slope < -0.01:
                    patterns.append(Pattern(
                        pattern_type="performance_degradation",
                        description="Performance degrades over time",
                        strength=min(1.0, abs(r_value)),
                        occurrences=len(scores),
                        examples=[f"Score trend: {slope:.3f} per evaluation"],
                        metadata={"slope": slope, "r_squared": r_value**2}
                    ))

        return patterns


class TrendAnalyzer:
    """Analyzes trends in evaluation metrics over time."""

    def __init__(self):
        """Initialize trend analyzer."""
        pass

    def analyze_metric_trends(self, results: list[EvaluationResult],
                            time_window_days: int = 30) -> list[Trend]:
        """Analyze trends in evaluation metrics over specified time window."""
        trends = []

        if len(results) < 5:
            return trends

        # Filter results to time window
        cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
        recent_results = [
            r for r in results
            if r.evaluated_at.replace(tzinfo=None) > cutoff_date
        ]

        if len(recent_results) < 3:
            return trends

        # Sort by time
        sorted_results = sorted(recent_results, key=lambda r: r.evaluated_at)

        # Metrics to analyze
        metric_extractors = {
            'overall_score': lambda r: r.metrics.overall_score,
            'skepticism_calibration': lambda r: r.metrics.skepticism_calibration,
            'evidence_standard_score': lambda r: r.metrics.evidence_standard_score,
            'red_flag_detection': lambda r: r.metrics.red_flag_detection,
            'reasoning_quality': lambda r: r.metrics.reasoning_quality,
            'confidence_level': lambda r: r.response.confidence_level
        }

        for metric_name, extractor in metric_extractors.items():
            values = [extractor(r) for r in sorted_results]
            time_indices = list(range(len(values)))

            if len(values) >= 3:
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_indices, values)

                # Determine trend direction
                if abs(slope) < 0.001:
                    direction = "stable"
                elif slope > 0:
                    direction = "increasing"
                else:
                    direction = "decreasing"

                # Calculate significance (strength of trend)
                significance = abs(r_value) if p_value < 0.05 else 0.0

                if significance > 0.3:  # Only include significant trends
                    trend = Trend(
                        metric_name=metric_name,
                        direction=direction,
                        slope=slope,
                        r_squared=r_value**2,
                        start_value=values[0],
                        end_value=values[-1],
                        time_period=f"{time_window_days} days",
                        significance=significance
                    )
                    trends.append(trend)

        return trends

    def detect_anomalies(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        """Detect anomalous evaluation results."""
        if len(results) < 10:
            return []

        # Extract overall scores
        scores = [r.metrics.overall_score for r in results]

        # Calculate Z-scores
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores)

        if std_score == 0:
            return []

        anomalies = []
        threshold = 2.5  # Z-score threshold for anomaly detection

        for result, score in zip(results, scores, strict=False):
            z_score = abs(score - mean_score) / std_score
            if z_score > threshold:
                anomalies.append(result)

        return anomalies
