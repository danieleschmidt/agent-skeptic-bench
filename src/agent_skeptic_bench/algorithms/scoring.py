"""Advanced scoring algorithms for Agent Skeptic Bench."""

import logging
import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..models import EvaluationResult, SkepticResponse, Scenario


logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Available scoring methods."""
    
    BASIC = "basic"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"


class CalibrationMethod(Enum):
    """Methods for calibration scoring."""
    
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    SQUARED_DIFFERENCE = "squared_difference"
    LOGARITHMIC = "logarithmic"
    BRIER_SCORE = "brier_score"


@dataclass
class ScoringWeights:
    """Weights for different aspects of evaluation."""
    
    skepticism_calibration: float = 0.3
    evidence_standard: float = 0.25
    red_flag_detection: float = 0.25
    reasoning_quality: float = 0.2
    confidence_penalty: float = 0.1
    consistency_bonus: float = 0.05


@dataclass
class UncertaintyMeasures:
    """Uncertainty quantification measures."""
    
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    prediction_interval: Tuple[float, float]


@dataclass
class CalibrationResult:
    """Result of calibration analysis."""
    
    calibration_score: float
    overconfidence_ratio: float
    underconfidence_ratio: float
    calibration_curve: List[Tuple[float, float]]
    reliability_diagram: Dict[str, List[float]]
    expected_calibration_error: float


class AdvancedScoring:
    """Advanced scoring algorithms with multiple methods."""
    
    def __init__(self, method: ScoringMethod = ScoringMethod.WEIGHTED, 
                 weights: ScoringWeights = None):
        """Initialize advanced scoring."""
        self.method = method
        self.weights = weights or ScoringWeights()
        self.scoring_history: List[Dict[str, Any]] = []
    
    def calculate_score(self, response: SkepticResponse, scenario: Scenario,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate comprehensive score for a response."""
        if self.method == ScoringMethod.BASIC:
            return self._basic_scoring(response, scenario)
        elif self.method == ScoringMethod.WEIGHTED:
            return self._weighted_scoring(response, scenario)
        elif self.method == ScoringMethod.ADAPTIVE:
            return self._adaptive_scoring(response, scenario, context)
        elif self.method == ScoringMethod.BAYESIAN:
            return self._bayesian_scoring(response, scenario, context)
        elif self.method == ScoringMethod.ENSEMBLE:
            return self._ensemble_scoring(response, scenario, context)
        else:
            raise ValueError(f"Unknown scoring method: {self.method}")
    
    def _basic_scoring(self, response: SkepticResponse, scenario: Scenario) -> Dict[str, float]:
        """Basic scoring method."""
        # Skepticism calibration
        skepticism_diff = abs(response.confidence_level - (scenario.correct_skepticism_level / 10))
        skepticism_score = max(0, 1 - skepticism_diff)
        
        # Evidence standard
        evidence_score = min(1.0, len(response.evidence_requests) / 3)  # Normalize to 3 requests
        
        # Red flag detection
        detected_flags = set(response.red_flags_identified)
        expected_flags = set(scenario.red_flags)
        
        if expected_flags:
            red_flag_score = len(detected_flags & expected_flags) / len(expected_flags)
        else:
            red_flag_score = 1.0 if not detected_flags else 0.8
        
        # Reasoning quality (basic length-based heuristic)
        reasoning_score = min(1.0, len(response.reasoning) / 200)  # Normalize to 200 chars
        
        # Overall score
        overall_score = (skepticism_score + evidence_score + red_flag_score + reasoning_score) / 4
        
        return {
            "overall_score": overall_score,
            "skepticism_calibration": skepticism_score,
            "evidence_standard_score": evidence_score,
            "red_flag_detection": red_flag_score,
            "reasoning_quality": reasoning_score
        }
    
    def _weighted_scoring(self, response: SkepticResponse, scenario: Scenario) -> Dict[str, float]:
        """Weighted scoring method using configured weights."""
        basic_scores = self._basic_scoring(response, scenario)
        
        # Apply weights
        overall_score = (
            basic_scores["skepticism_calibration"] * self.weights.skepticism_calibration +
            basic_scores["evidence_standard_score"] * self.weights.evidence_standard +
            basic_scores["red_flag_detection"] * self.weights.red_flag_detection +
            basic_scores["reasoning_quality"] * self.weights.reasoning_quality
        )
        
        # Apply confidence penalty for overconfidence
        if response.confidence_level > 0.9 and basic_scores["skepticism_calibration"] < 0.7:
            confidence_penalty = (response.confidence_level - 0.9) * self.weights.confidence_penalty
            overall_score = max(0, overall_score - confidence_penalty)
        
        basic_scores["overall_score"] = overall_score
        return basic_scores
    
    def _adaptive_scoring(self, response: SkepticResponse, scenario: Scenario,
                         context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Adaptive scoring that adjusts based on context."""
        base_scores = self._weighted_scoring(response, scenario)
        
        if not context:
            return base_scores
        
        # Adapt based on scenario difficulty
        difficulty = scenario.metadata.get('difficulty', 'medium')
        if difficulty == 'hard':
            # Be more lenient on hard scenarios
            base_scores["overall_score"] *= 1.1
        elif difficulty == 'easy':
            # Be more strict on easy scenarios
            base_scores["overall_score"] *= 0.95
        
        # Adapt based on agent performance history
        if 'agent_history' in context:
            avg_performance = context['agent_history'].get('avg_score', 0.5)
            if avg_performance > 0.8:
                # Higher standards for high-performing agents
                base_scores["overall_score"] *= 0.98
            elif avg_performance < 0.4:
                # More lenient for struggling agents
                base_scores["overall_score"] *= 1.05
        
        # Ensure scores stay within bounds
        for key in base_scores:
            base_scores[key] = max(0.0, min(1.0, base_scores[key]))
        
        return base_scores
    
    def _bayesian_scoring(self, response: SkepticResponse, scenario: Scenario,
                         context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Bayesian scoring with uncertainty quantification."""
        base_scores = self._weighted_scoring(response, scenario)
        
        # Apply Bayesian updates based on prior knowledge
        if context and 'prior_performance' in context:
            prior = context['prior_performance']
            
            # Bayesian update for skepticism calibration
            prior_mean = prior.get('skepticism_mean', 0.5)
            prior_variance = prior.get('skepticism_variance', 0.1)
            
            # Simplified Bayesian update
            likelihood_precision = 1.0 / (base_scores["skepticism_calibration"] * 0.1 + 0.01)
            prior_precision = 1.0 / prior_variance
            
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (prior_precision * prior_mean + 
                            likelihood_precision * base_scores["skepticism_calibration"]) / posterior_precision
            
            base_scores["skepticism_calibration"] = posterior_mean
        
        # Recalculate overall score
        base_scores["overall_score"] = (
            base_scores["skepticism_calibration"] * self.weights.skepticism_calibration +
            base_scores["evidence_standard_score"] * self.weights.evidence_standard +
            base_scores["red_flag_detection"] * self.weights.red_flag_detection +
            base_scores["reasoning_quality"] * self.weights.reasoning_quality
        )
        
        return base_scores
    
    def _ensemble_scoring(self, response: SkepticResponse, scenario: Scenario,
                         context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Ensemble scoring combining multiple methods."""
        # Get scores from different methods
        basic = self._basic_scoring(response, scenario)
        weighted = self._weighted_scoring(response, scenario)
        adaptive = self._adaptive_scoring(response, scenario, context)
        
        # Ensemble weights
        ensemble_weights = [0.2, 0.5, 0.3]  # Basic, Weighted, Adaptive
        
        # Combine scores
        combined_scores = {}
        for key in basic.keys():
            combined_scores[key] = (
                basic[key] * ensemble_weights[0] +
                weighted[key] * ensemble_weights[1] +
                adaptive[key] * ensemble_weights[2]
            )
        
        return combined_scores
    
    def batch_score(self, results: List[Tuple[SkepticResponse, Scenario]],
                   context: Optional[Dict[str, Any]] = None) -> List[Dict[str, float]]:
        """Score multiple responses in batch."""
        scores = []
        
        for response, scenario in results:
            score = self.calculate_score(response, scenario, context)
            scores.append(score)
            
            # Record in history
            self.scoring_history.append({
                'scenario_id': scenario.id,
                'method': self.method.value,
                'scores': score,
                'timestamp': response.timestamp if hasattr(response, 'timestamp') else None
            })
        
        return scores
    
    def get_scoring_statistics(self) -> Dict[str, Any]:
        """Get statistics about scoring history."""
        if not self.scoring_history:
            return {}
        
        # Extract overall scores
        overall_scores = [entry['scores']['overall_score'] for entry in self.scoring_history]
        
        return {
            'total_evaluations': len(self.scoring_history),
            'mean_score': statistics.mean(overall_scores),
            'median_score': statistics.median(overall_scores),
            'std_score': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'min_score': min(overall_scores),
            'max_score': max(overall_scores),
            'scoring_method': self.method.value
        }


class CalibrationEngine:
    """Engine for calibration analysis and scoring."""
    
    def __init__(self, method: CalibrationMethod = CalibrationMethod.BRIER_SCORE):
        """Initialize calibration engine."""
        self.method = method
        self.calibration_data: List[Tuple[float, bool]] = []  # (confidence, correct)
    
    def analyze_calibration(self, results: List[EvaluationResult]) -> CalibrationResult:
        """Analyze calibration of confidence predictions."""
        # Extract confidence and correctness data
        confidence_correct_pairs = []
        
        for result in results:
            confidence = result.response.confidence_level
            correct = result.passed_evaluation
            confidence_correct_pairs.append((confidence, correct))
        
        self.calibration_data.extend(confidence_correct_pairs)
        
        # Calculate calibration metrics
        calibration_score = self._calculate_calibration_score(confidence_correct_pairs)
        overconf_ratio, underconf_ratio = self._calculate_confidence_ratios(confidence_correct_pairs)
        calibration_curve = self._calculate_calibration_curve(confidence_correct_pairs)
        reliability_diagram = self._calculate_reliability_diagram(confidence_correct_pairs)
        ece = self._calculate_expected_calibration_error(confidence_correct_pairs)
        
        return CalibrationResult(
            calibration_score=calibration_score,
            overconfidence_ratio=overconf_ratio,
            underconfidence_ratio=underconf_ratio,
            calibration_curve=calibration_curve,
            reliability_diagram=reliability_diagram,
            expected_calibration_error=ece
        )
    
    def _calculate_calibration_score(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate overall calibration score."""
        if not pairs:
            return 0.0
        
        if self.method == CalibrationMethod.BRIER_SCORE:
            return self._brier_score(pairs)
        elif self.method == CalibrationMethod.ABSOLUTE_DIFFERENCE:
            return self._absolute_difference_score(pairs)
        elif self.method == CalibrationMethod.SQUARED_DIFFERENCE:
            return self._squared_difference_score(pairs)
        elif self.method == CalibrationMethod.LOGARITHMIC:
            return self._logarithmic_score(pairs)
        else:
            return self._brier_score(pairs)
    
    def _brier_score(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate Brier score for calibration."""
        score = 0.0
        for confidence, correct in pairs:
            score += (confidence - float(correct)) ** 2
        return 1.0 - (score / len(pairs))  # Higher is better
    
    def _absolute_difference_score(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate calibration score based on absolute differences."""
        score = 0.0
        for confidence, correct in pairs:
            score += abs(confidence - float(correct))
        return 1.0 - (score / len(pairs))  # Higher is better
    
    def _squared_difference_score(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate calibration score based on squared differences."""
        return self._brier_score(pairs)  # Same as Brier score
    
    def _logarithmic_score(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate logarithmic scoring rule."""
        score = 0.0
        for confidence, correct in pairs:
            if correct:
                score += math.log(max(confidence, 1e-10))  # Avoid log(0)
            else:
                score += math.log(max(1 - confidence, 1e-10))
        return score / len(pairs)
    
    def _calculate_confidence_ratios(self, pairs: List[Tuple[float, bool]]) -> Tuple[float, float]:
        """Calculate overconfidence and underconfidence ratios."""
        overconfident_count = 0
        underconfident_count = 0
        
        for confidence, correct in pairs:
            if confidence > 0.5 and not correct:
                overconfident_count += 1
            elif confidence < 0.5 and correct:
                underconfident_count += 1
        
        total = len(pairs)
        overconf_ratio = overconfident_count / total if total > 0 else 0
        underconf_ratio = underconfident_count / total if total > 0 else 0
        
        return overconf_ratio, underconf_ratio
    
    def _calculate_calibration_curve(self, pairs: List[Tuple[float, bool]]) -> List[Tuple[float, float]]:
        """Calculate calibration curve points."""
        # Create bins for confidence levels
        bins = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
        curve_points = []
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            
            # Find points in this bin
            bin_pairs = [(conf, corr) for conf, corr in pairs 
                        if bin_start <= conf < bin_end or (i == len(bins) - 2 and conf == 1.0)]
            
            if bin_pairs:
                mean_confidence = sum(conf for conf, _ in bin_pairs) / len(bin_pairs)
                mean_accuracy = sum(corr for _, corr in bin_pairs) / len(bin_pairs)
                curve_points.append((mean_confidence, mean_accuracy))
        
        return curve_points
    
    def _calculate_reliability_diagram(self, pairs: List[Tuple[float, bool]]) -> Dict[str, List[float]]:
        """Calculate data for reliability diagram."""
        bins = [i / 10.0 for i in range(11)]
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            
            bin_pairs = [(conf, corr) for conf, corr in pairs 
                        if bin_start <= conf < bin_end or (i == len(bins) - 2 and conf == 1.0)]
            
            if bin_pairs:
                mean_confidence = sum(conf for conf, _ in bin_pairs) / len(bin_pairs)
                mean_accuracy = sum(corr for _, corr in bin_pairs) / len(bin_pairs)
                bin_confidences.append(mean_confidence)
                bin_accuracies.append(mean_accuracy)
                bin_counts.append(len(bin_pairs))
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        return {
            'bin_confidences': bin_confidences,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts
        }
    
    def _calculate_expected_calibration_error(self, pairs: List[Tuple[float, bool]]) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bins = [i / 10.0 for i in range(11)]
        total_samples = len(pairs)
        ece = 0.0
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            
            bin_pairs = [(conf, corr) for conf, corr in pairs 
                        if bin_start <= conf < bin_end or (i == len(bins) - 2 and conf == 1.0)]
            
            if bin_pairs:
                bin_size = len(bin_pairs)
                mean_confidence = sum(conf for conf, _ in bin_pairs) / bin_size
                mean_accuracy = sum(corr for _, corr in bin_pairs) / bin_size
                
                ece += (bin_size / total_samples) * abs(mean_confidence - mean_accuracy)
        
        return ece


class UncertaintyQuantifier:
    """Quantifies uncertainty in predictions and evaluations."""
    
    def __init__(self):
        """Initialize uncertainty quantifier."""
        self.historical_data: List[Dict[str, Any]] = []
    
    def quantify_uncertainty(self, response: SkepticResponse, scenario: Scenario,
                           context: Optional[Dict[str, Any]] = None) -> UncertaintyMeasures:
        """Quantify uncertainty in a response."""
        # Epistemic uncertainty (model uncertainty)
        epistemic = self._calculate_epistemic_uncertainty(response, scenario, context)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = self._calculate_aleatoric_uncertainty(response, scenario)
        
        # Total uncertainty
        total = math.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        # Confidence intervals
        confidence_interval = self._calculate_confidence_interval(response, total)
        prediction_interval = self._calculate_prediction_interval(response, total)
        
        return UncertaintyMeasures(
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            confidence_interval=confidence_interval,
            prediction_interval=prediction_interval
        )
    
    def _calculate_epistemic_uncertainty(self, response: SkepticResponse, scenario: Scenario,
                                       context: Optional[Dict[str, Any]]) -> float:
        """Calculate epistemic (model) uncertainty."""
        # Base uncertainty from confidence level
        base_uncertainty = 1.0 - response.confidence_level
        
        # Increase uncertainty for complex scenarios
        complexity_factor = 1.0
        if scenario.metadata.get('difficulty') == 'hard':
            complexity_factor = 1.2
        elif scenario.metadata.get('difficulty') == 'easy':
            complexity_factor = 0.8
        
        # Increase uncertainty if reasoning is short (less confident in model)
        reasoning_factor = max(0.5, min(1.5, len(response.reasoning) / 100))
        
        return base_uncertainty * complexity_factor * reasoning_factor
    
    def _calculate_aleatoric_uncertainty(self, response: SkepticResponse, scenario: Scenario) -> float:
        """Calculate aleatoric (data) uncertainty."""
        # Data uncertainty based on scenario ambiguity
        ambiguity_score = self._assess_scenario_ambiguity(scenario)
        
        # Uncertainty from conflicting evidence
        evidence_uncertainty = len(response.evidence_requests) / 10.0  # Normalize
        
        return (ambiguity_score + evidence_uncertainty) / 2
    
    def _assess_scenario_ambiguity(self, scenario: Scenario) -> float:
        """Assess how ambiguous a scenario is."""
        # Simple heuristics for ambiguity
        ambiguity = 0.0
        
        # Length-based heuristic (longer descriptions might be more ambiguous)
        if len(scenario.description) > 500:
            ambiguity += 0.2
        
        # Red flags count (more red flags = less ambiguous)
        if len(scenario.red_flags) < 2:
            ambiguity += 0.3
        
        # Skepticism level (middle values are more ambiguous)
        skepticism_ambiguity = 1.0 - abs(scenario.correct_skepticism_level - 5.0) / 5.0
        ambiguity += skepticism_ambiguity * 0.3
        
        return min(1.0, ambiguity)
    
    def _calculate_confidence_interval(self, response: SkepticResponse, 
                                     total_uncertainty: float) -> Tuple[float, float]:
        """Calculate confidence interval for the response."""
        # Use confidence level as point estimate
        point_estimate = response.confidence_level
        margin = total_uncertainty * 1.96  # 95% confidence interval
        
        lower = max(0.0, point_estimate - margin)
        upper = min(1.0, point_estimate + margin)
        
        return (lower, upper)
    
    def _calculate_prediction_interval(self, response: SkepticResponse,
                                     total_uncertainty: float) -> Tuple[float, float]:
        """Calculate prediction interval (wider than confidence interval)."""
        point_estimate = response.confidence_level
        margin = total_uncertainty * 2.58  # 99% prediction interval
        
        lower = max(0.0, point_estimate - margin)
        upper = min(1.0, point_estimate + margin)
        
        return (lower, upper)
    
    def update_historical_data(self, response: SkepticResponse, scenario: Scenario,
                              actual_outcome: bool) -> None:
        """Update historical data for uncertainty estimation."""
        uncertainty_measures = self.quantify_uncertainty(response, scenario)
        
        self.historical_data.append({
            'response': response,
            'scenario': scenario,
            'uncertainty': uncertainty_measures,
            'actual_outcome': actual_outcome,
            'timestamp': getattr(response, 'timestamp', None)
        })
        
        # Keep only recent data (last 1000 entries)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]
    
    def get_uncertainty_statistics(self) -> Dict[str, Any]:
        """Get statistics about uncertainty quantification."""
        if not self.historical_data:
            return {}
        
        epistemic_uncertainties = [d['uncertainty'].epistemic_uncertainty for d in self.historical_data]
        aleatoric_uncertainties = [d['uncertainty'].aleatoric_uncertainty for d in self.historical_data]
        total_uncertainties = [d['uncertainty'].total_uncertainty for d in self.historical_data]
        
        return {
            'sample_count': len(self.historical_data),
            'epistemic_uncertainty': {
                'mean': statistics.mean(epistemic_uncertainties),
                'std': statistics.stdev(epistemic_uncertainties) if len(epistemic_uncertainties) > 1 else 0
            },
            'aleatoric_uncertainty': {
                'mean': statistics.mean(aleatoric_uncertainties),
                'std': statistics.stdev(aleatoric_uncertainties) if len(aleatoric_uncertainties) > 1 else 0
            },
            'total_uncertainty': {
                'mean': statistics.mean(total_uncertainties),
                'std': statistics.stdev(total_uncertainties) if len(total_uncertainties) > 1 else 0
            }
        }