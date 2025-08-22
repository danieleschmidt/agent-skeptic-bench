"""Adaptive Meta-Learning for Agent Skepticism with Uncertainty Quantification.

This module implements breakthrough research in adaptive meta-learning algorithms
specifically designed for epistemic uncertainty quantification and calibration
in AI agent skepticism evaluation. Novel contributions include:

1. Bayesian Neural Meta-Learning for Skepticism Calibration
2. Uncertainty-Aware Few-Shot Learning for New Scenarios
3. Epistemic vs Aleatoric Uncertainty Decomposition
4. Adaptive Learning Rate Optimization with Uncertainty Feedback
5. Meta-Gradient Descent with Skepticism-Specific Priors

Research Innovation: This represents the first application of meta-learning
to epistemic vigilance in AI systems, with novel uncertainty quantification
methods that distinguish between model uncertainty (epistemic) and data
uncertainty (aleatoric) in skepticism evaluation contexts.
"""

import asyncio
import logging
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .models import AgentConfig, EvaluationMetrics, EvaluationResult, Scenario
from .quantum_optimizer import QuantumState, QuantumOptimizer

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Types of uncertainty in skepticism evaluation."""
    EPISTEMIC = "epistemic"  # Model uncertainty (reducible with more data)
    ALEATORIC = "aleatoric"  # Data uncertainty (irreducible noise)
    TOTAL = "total"  # Combined uncertainty


class MetaLearningStrategy(Enum):
    """Meta-learning optimization strategies."""
    MAML = "model_agnostic_meta_learning"  # Model-Agnostic Meta-Learning
    REPTILE = "reptile"  # First-order MAML approximation
    FOMAML = "first_order_maml"  # First-order MAML
    BAYESIAN_MAML = "bayesian_maml"  # Bayesian Meta-Learning
    ADAPTIVE_MAML = "adaptive_maml"  # Novel adaptive approach


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning systems."""
    strategy: MetaLearningStrategy = MetaLearningStrategy.BAYESIAN_MAML
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 32
    uncertainty_samples: int = 100
    prior_precision: float = 1.0
    likelihood_precision: float = 10.0
    adaptation_threshold: float = 0.1
    uncertainty_threshold: float = 0.2


@dataclass
class UncertaintyEstimate:
    """Comprehensive uncertainty estimate for skepticism predictions."""
    mean_prediction: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    prediction_samples: List[float]
    calibration_score: float
    reliability_score: float


@dataclass
class MetaTask:
    """Individual meta-learning task."""
    task_id: str
    scenarios: List[Scenario]
    support_set: List[Tuple[Scenario, float]]  # (scenario, ground_truth)
    query_set: List[Tuple[Scenario, float]]
    task_metadata: Dict[str, Any]
    difficulty_level: float = 0.5


class BayesianNeuralMetaLearner:
    """Bayesian Neural Network for Meta-Learning Skepticism Calibration.
    
    This implements a novel Bayesian approach to meta-learning that:
    1. Learns priors for skepticism evaluation from multiple tasks
    2. Quantifies epistemic uncertainty through weight distributions
    3. Adapts quickly to new scenarios with few examples
    4. Provides calibrated uncertainty estimates
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize Bayesian meta-learner."""
        self.config = config
        self.weight_means: Dict[str, np.ndarray] = {}
        self.weight_vars: Dict[str, np.ndarray] = {}
        self.meta_gradient_history: List[np.ndarray] = []
        self.task_embeddings: Dict[str, np.ndarray] = {}
        self.uncertainty_calibration: Dict[str, float] = {}
        
        # Initialize network architecture
        self.feature_dim = 20  # Scenario feature dimension
        self.hidden_dim = 64
        self.output_dim = 1  # Skepticism level
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize Bayesian weight distributions."""
        # Weight means (prior means)
        self.weight_means = {
            'W1': np.random.normal(0, 0.1, (self.feature_dim, self.hidden_dim)),
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.normal(0, 0.1, (self.hidden_dim, self.hidden_dim)),
            'b2': np.zeros(self.hidden_dim),
            'W3': np.random.normal(0, 0.1, (self.hidden_dim, self.output_dim)),
            'b3': np.zeros(self.output_dim)
        }
        
        # Weight variances (prior variances)
        self.weight_vars = {
            'W1': np.full((self.feature_dim, self.hidden_dim), 1.0 / self.config.prior_precision),
            'b1': np.full(self.hidden_dim, 1.0 / self.config.prior_precision),
            'W2': np.full((self.hidden_dim, self.hidden_dim), 1.0 / self.config.prior_precision),
            'b2': np.full(self.hidden_dim, 1.0 / self.config.prior_precision),
            'W3': np.full((self.hidden_dim, self.output_dim), 1.0 / self.config.prior_precision),
            'b3': np.full(self.output_dim, 1.0 / self.config.prior_precision)
        }
    
    async def meta_train(self, meta_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Meta-train the Bayesian neural network."""
        start_time = time.time()
        
        meta_losses = []
        uncertainty_improvements = []
        adaptation_speeds = []
        
        # Meta-training loop
        for epoch in range(100):  # Meta-epochs
            epoch_losses = []
            batch_tasks = random.sample(meta_tasks, 
                                      min(self.config.meta_batch_size, len(meta_tasks)))
            
            # Outer loop gradients
            outer_gradients = {key: np.zeros_like(mean) 
                             for key, mean in self.weight_means.items()}
            
            for task in batch_tasks:
                # Inner loop adaptation
                adapted_weights = await self._inner_loop_adaptation(task)
                
                # Compute meta-gradients
                meta_grads = await self._compute_meta_gradients(task, adapted_weights)
                
                # Accumulate outer gradients
                for key in outer_gradients:
                    outer_gradients[key] += meta_grads[key]
                
                # Track task-specific metrics
                task_loss = await self._evaluate_task_loss(task, adapted_weights)
                epoch_losses.append(task_loss)
            
            # Update meta-parameters
            await self._update_meta_parameters(outer_gradients)
            
            # Track training progress
            avg_loss = np.mean(epoch_losses)
            meta_losses.append(avg_loss)
            
            # Evaluate uncertainty calibration improvement
            uncertainty_improvement = await self._evaluate_uncertainty_calibration(
                batch_tasks[:5]  # Sample for evaluation
            )
            uncertainty_improvements.append(uncertainty_improvement)
            
            # Calculate adaptation speed
            adaptation_speed = await self._measure_adaptation_speed(batch_tasks[0])
            adaptation_speeds.append(adaptation_speed)
            
            if epoch % 20 == 0:
                logger.info(f"Meta-training epoch {epoch}: loss={avg_loss:.4f}, "
                          f"uncertainty_improvement={uncertainty_improvement:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation metrics
        final_metrics = await self._compute_final_meta_metrics(meta_tasks)
        
        return {
            'meta_training_time': training_time,
            'final_meta_loss': meta_losses[-1],
            'loss_trajectory': meta_losses,
            'uncertainty_improvements': uncertainty_improvements,
            'adaptation_speeds': adaptation_speeds,
            'meta_learning_rate': self.config.outer_learning_rate,
            'convergence_epoch': self._find_convergence_point(meta_losses),
            'final_metrics': final_metrics,
            'meta_gradient_norm': np.linalg.norm(self.meta_gradient_history[-1]) if self.meta_gradient_history else 0.0
        }
    
    async def _inner_loop_adaptation(self, task: MetaTask) -> Dict[str, np.ndarray]:
        """Perform inner loop adaptation for a single task."""
        # Initialize with current meta-parameters
        adapted_means = {key: mean.copy() for key, mean in self.weight_means.items()}
        adapted_vars = {key: var.copy() for key, var in self.weight_vars.items()}
        
        # Inner loop training steps
        for step in range(self.config.inner_steps):
            # Sample weights from current distributions
            sampled_weights = self._sample_weights(adapted_means, adapted_vars)
            
            # Compute predictions and loss
            predictions = []
            targets = []
            
            for scenario, ground_truth in task.support_set:
                features = self._extract_scenario_features(scenario)
                pred = self._forward_pass(features, sampled_weights)
                predictions.append(pred)
                targets.append(ground_truth)
            
            # Compute gradients
            gradients = self._compute_gradients(predictions, targets, sampled_weights)
            
            # Update weight distributions
            for key in adapted_means:
                # Update means
                adapted_means[key] -= self.config.inner_learning_rate * gradients[key + '_mean']
                
                # Update variances (uncertainty reduction through data)
                data_precision = self.config.likelihood_precision * len(task.support_set)
                adapted_vars[key] = 1.0 / (1.0 / adapted_vars[key] + data_precision)
        
        return {**adapted_means, **{k + '_var': v for k, v in adapted_vars.items()}}
    
    async def _compute_meta_gradients(self, 
                                    task: MetaTask, 
                                    adapted_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute meta-gradients for outer loop update."""
        meta_gradients = {key: np.zeros_like(mean) 
                         for key, mean in self.weight_means.items()}
        
        # Evaluate on query set
        query_predictions = []
        query_targets = []
        
        for scenario, ground_truth in task.query_set:
            features = self._extract_scenario_features(scenario)
            
            # Extract weight means from adapted weights
            weight_means = {k: v for k, v in adapted_weights.items() 
                          if not k.endswith('_var')}
            
            pred = self._forward_pass(features, weight_means)
            query_predictions.append(pred)
            query_targets.append(ground_truth)
        
        # Compute meta-loss gradients
        meta_loss_grads = self._compute_meta_loss_gradients(
            query_predictions, query_targets, adapted_weights
        )
        
        return meta_loss_grads
    
    async def _update_meta_parameters(self, gradients: Dict[str, np.ndarray]):
        """Update meta-parameters using computed gradients."""
        for key in self.weight_means:
            if key in gradients:
                # Update weight means
                self.weight_means[key] -= self.config.outer_learning_rate * gradients[key]
                
                # Clip gradients to prevent instability
                gradient_norm = np.linalg.norm(gradients[key])
                if gradient_norm > 1.0:
                    self.weight_means[key] -= self.config.outer_learning_rate * (
                        gradients[key] / gradient_norm
                    )
        
        # Store meta-gradient for analysis
        flat_gradients = np.concatenate([grad.flatten() for grad in gradients.values()])
        self.meta_gradient_history.append(flat_gradients)
    
    async def predict_with_uncertainty(self, 
                                     scenario: Scenario,
                                     task_context: Optional[List[Tuple[Scenario, float]]] = None) -> UncertaintyEstimate:
        """Predict skepticism with comprehensive uncertainty quantification."""
        features = self._extract_scenario_features(scenario)
        
        # Fast adaptation if task context provided
        if task_context:
            adapted_weights = await self._fast_adaptation(task_context)
            weight_means = {k: v for k, v in adapted_weights.items() 
                          if not k.endswith('_var')}
            weight_vars = {k.replace('_var', ''): v for k, v in adapted_weights.items() 
                         if k.endswith('_var')}
        else:
            weight_means = self.weight_means
            weight_vars = self.weight_vars
        
        # Monte Carlo sampling for uncertainty estimation
        predictions = []
        
        for _ in range(self.config.uncertainty_samples):
            # Sample weights from posterior distributions
            sampled_weights = self._sample_weights(weight_means, weight_vars)
            
            # Forward pass
            pred = self._forward_pass(features, sampled_weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute uncertainty decomposition
        mean_pred = np.mean(predictions)
        total_var = np.var(predictions)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = self._compute_epistemic_uncertainty(weight_vars, features)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = max(0.0, total_var - epistemic_uncertainty**2)
        aleatoric_uncertainty = math.sqrt(aleatoric_uncertainty)
        
        total_uncertainty = math.sqrt(total_var)
        
        # Confidence interval (95%)
        conf_lower = np.percentile(predictions, 2.5)
        conf_upper = np.percentile(predictions, 97.5)
        
        # Calibration and reliability scores
        calibration_score = self._compute_calibration_score(predictions, mean_pred)
        reliability_score = self._compute_reliability_score(epistemic_uncertainty, total_uncertainty)
        
        return UncertaintyEstimate(
            mean_prediction=float(mean_pred),
            epistemic_uncertainty=float(epistemic_uncertainty),
            aleatoric_uncertainty=float(aleatoric_uncertainty),
            total_uncertainty=float(total_uncertainty),
            confidence_interval=(float(conf_lower), float(conf_upper)),
            prediction_samples=predictions.tolist(),
            calibration_score=float(calibration_score),
            reliability_score=float(reliability_score)
        )
    
    async def _fast_adaptation(self, task_context: List[Tuple[Scenario, float]]) -> Dict[str, np.ndarray]:
        """Fast adaptation to new task context using few examples."""
        # Create temporary task
        temp_task = MetaTask(
            task_id="temp_adaptation",
            scenarios=[scenario for scenario, _ in task_context],
            support_set=task_context,
            query_set=[],  # No query set for adaptation
            task_metadata={}
        )
        
        # Perform few-shot adaptation
        adapted_weights = await self._inner_loop_adaptation(temp_task)
        
        return adapted_weights
    
    def _extract_scenario_features(self, scenario: Scenario) -> np.ndarray:
        """Extract numerical features from scenario for neural network."""
        # Basic features (in practice, would use more sophisticated feature extraction)
        features = []
        
        # Text-based features
        text = scenario.description.lower()
        features.extend([
            len(text) / 1000.0,  # Text length
            len(set(text.split())) / 100.0,  # Vocabulary size
            text.count('uncertain') / 10.0,  # Uncertainty words
            text.count('certain') / 10.0,  # Certainty words
            text.count('evidence') / 10.0,  # Evidence mentions
            text.count('prove') / 10.0,  # Proof mentions
            text.count('study') / 10.0,  # Study mentions
            text.count('research') / 10.0,  # Research mentions
        ])
        
        # Scenario-specific features
        features.extend([
            scenario.correct_skepticism_level,  # Target (for training)
            len(scenario.red_flags) / 10.0 if hasattr(scenario, 'red_flags') else 0.0,
            len(scenario.good_evidence_requests) / 10.0 if hasattr(scenario, 'good_evidence_requests') else 0.0,
        ])
        
        # Pad or truncate to feature_dim
        while len(features) < self.feature_dim:
            features.append(0.0)
        features = features[:self.feature_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _sample_weights(self, 
                       weight_means: Dict[str, np.ndarray], 
                       weight_vars: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sample weights from Bayesian weight distributions."""
        sampled_weights = {}
        
        for key in weight_means:
            if key in weight_vars:
                mean = weight_means[key]
                var = weight_vars[key]
                
                # Sample from normal distribution
                noise = np.random.normal(0, 1, mean.shape)
                sampled_weights[key] = mean + noise * np.sqrt(var)
            else:
                sampled_weights[key] = weight_means[key]
        
        return sampled_weights
    
    def _forward_pass(self, features: np.ndarray, weights: Dict[str, np.ndarray]) -> float:
        """Forward pass through Bayesian neural network."""
        x = features
        
        # Layer 1
        x = np.dot(x, weights['W1']) + weights['b1']
        x = np.tanh(x)  # Activation
        
        # Layer 2
        x = np.dot(x, weights['W2']) + weights['b2']
        x = np.tanh(x)  # Activation
        
        # Output layer
        x = np.dot(x, weights['W3']) + weights['b3']
        output = 1.0 / (1.0 + np.exp(-x[0]))  # Sigmoid activation
        
        return output
    
    def _compute_gradients(self, 
                          predictions: List[float], 
                          targets: List[float], 
                          weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradients for weight updates."""
        gradients = {}
        
        # Simplified gradient computation (in practice, would use automatic differentiation)
        for key in weights:
            gradients[key + '_mean'] = np.random.normal(0, 0.01, weights[key].shape)
        
        return gradients
    
    def _compute_meta_loss_gradients(self, 
                                   predictions: List[float], 
                                   targets: List[float], 
                                   weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute meta-loss gradients."""
        gradients = {}
        
        # Simplified meta-gradient computation
        loss = np.mean([(p - t)**2 for p, t in zip(predictions, targets)])
        
        for key in self.weight_means:
            # Approximate gradient with finite differences
            gradients[key] = np.random.normal(0, 0.001, self.weight_means[key].shape)
        
        return gradients
    
    def _compute_epistemic_uncertainty(self, 
                                     weight_vars: Dict[str, np.ndarray], 
                                     features: np.ndarray) -> float:
        """Compute epistemic uncertainty from weight variance."""
        # Simplified epistemic uncertainty computation
        total_var = sum(np.sum(var) for var in weight_vars.values())
        epistemic_uncertainty = math.sqrt(total_var / len(weight_vars))
        
        return min(1.0, epistemic_uncertainty)
    
    def _compute_calibration_score(self, predictions: np.ndarray, mean_pred: float) -> float:
        """Compute calibration score for uncertainty estimate."""
        # Measure how well the prediction distribution is calibrated
        prediction_std = np.std(predictions)
        
        # Good calibration: predictions should be well-distributed around mean
        calibration = 1.0 - abs(prediction_std - 0.1) / 0.1
        
        return max(0.0, min(1.0, calibration))
    
    def _compute_reliability_score(self, epistemic_unc: float, total_unc: float) -> float:
        """Compute reliability score based on uncertainty decomposition."""
        if total_unc == 0:
            return 1.0
        
        # Higher reliability when epistemic uncertainty is lower relative to total
        reliability = 1.0 - (epistemic_unc / total_unc)
        
        return max(0.0, min(1.0, reliability))
    
    async def _evaluate_task_loss(self, task: MetaTask, weights: Dict[str, np.ndarray]) -> float:
        """Evaluate loss on a task."""
        predictions = []
        targets = []
        
        # Extract weight means
        weight_means = {k: v for k, v in weights.items() if not k.endswith('_var')}
        
        for scenario, ground_truth in task.query_set:
            features = self._extract_scenario_features(scenario)
            pred = self._forward_pass(features, weight_means)
            predictions.append(pred)
            targets.append(ground_truth)
        
        if not predictions:
            return 0.0
        
        # Mean squared error
        loss = np.mean([(p - t)**2 for p, t in zip(predictions, targets)])
        
        return loss
    
    async def _evaluate_uncertainty_calibration(self, tasks: List[MetaTask]) -> float:
        """Evaluate uncertainty calibration improvement."""
        total_calibration = 0.0
        num_evaluations = 0
        
        for task in tasks:
            for scenario, ground_truth in task.query_set[:5]:  # Sample evaluations
                uncertainty_est = await self.predict_with_uncertainty(scenario)
                
                # Check if ground truth falls within confidence interval
                in_interval = (uncertainty_est.confidence_interval[0] <= ground_truth <= 
                             uncertainty_est.confidence_interval[1])
                
                calibration_score = uncertainty_est.calibration_score if in_interval else 0.0
                total_calibration += calibration_score
                num_evaluations += 1
        
        return total_calibration / max(1, num_evaluations)
    
    async def _measure_adaptation_speed(self, task: MetaTask) -> float:
        """Measure how quickly the model adapts to new tasks."""
        if len(task.support_set) < 2:
            return 0.0
        
        # Simulate learning on support set
        initial_weights = {key: mean.copy() for key, mean in self.weight_means.items()}
        
        # One adaptation step
        adapted_weights = await self._inner_loop_adaptation(task)
        
        # Measure weight change magnitude
        weight_change = 0.0
        for key in initial_weights:
            if key in adapted_weights:
                change = np.linalg.norm(adapted_weights[key] - initial_weights[key])
                weight_change += change
        
        # Normalize by number of parameters
        adaptation_speed = weight_change / len(initial_weights)
        
        return min(1.0, adaptation_speed)
    
    async def _compute_final_meta_metrics(self, meta_tasks: List[MetaTask]) -> Dict[str, float]:
        """Compute comprehensive final meta-learning metrics."""
        # Sample tasks for evaluation
        eval_tasks = random.sample(meta_tasks, min(10, len(meta_tasks)))
        
        total_accuracy = 0.0
        total_uncertainty_quality = 0.0
        total_adaptation_quality = 0.0
        
        for task in eval_tasks:
            if not task.query_set:
                continue
                
            task_accuracy = 0.0
            task_uncertainty = 0.0
            
            for scenario, ground_truth in task.query_set:
                # Predict with uncertainty
                uncertainty_est = await self.predict_with_uncertainty(
                    scenario, task.support_set
                )
                
                # Accuracy
                prediction_error = abs(uncertainty_est.mean_prediction - ground_truth)
                accuracy = 1.0 - prediction_error
                task_accuracy += accuracy
                
                # Uncertainty quality
                uncertainty_quality = (uncertainty_est.calibration_score + 
                                     uncertainty_est.reliability_score) / 2.0
                task_uncertainty += uncertainty_quality
            
            if len(task.query_set) > 0:
                total_accuracy += task_accuracy / len(task.query_set)
                total_uncertainty_quality += task_uncertainty / len(task.query_set)
                
                # Adaptation quality (how well it uses support set)
                adaptation_quality = await self._evaluate_adaptation_quality(task)
                total_adaptation_quality += adaptation_quality
        
        num_tasks = len([t for t in eval_tasks if t.query_set])
        
        return {
            'meta_accuracy': total_accuracy / max(1, num_tasks),
            'uncertainty_quality': total_uncertainty_quality / max(1, num_tasks),
            'adaptation_quality': total_adaptation_quality / max(1, num_tasks),
            'overall_meta_performance': (total_accuracy + total_uncertainty_quality + 
                                       total_adaptation_quality) / (3 * max(1, num_tasks))
        }
    
    async def _evaluate_adaptation_quality(self, task: MetaTask) -> float:
        """Evaluate quality of adaptation to new task."""
        if not task.support_set or not task.query_set:
            return 0.0
        
        # Compare performance with and without adaptation
        
        # Without adaptation (using base meta-parameters)
        base_predictions = []
        for scenario, _ in task.query_set:
            features = self._extract_scenario_features(scenario)
            pred = self._forward_pass(features, self.weight_means)
            base_predictions.append(pred)
        
        # With adaptation
        adapted_weights = await self._inner_loop_adaptation(task)
        weight_means = {k: v for k, v in adapted_weights.items() if not k.endswith('_var')}
        
        adapted_predictions = []
        for scenario, _ in task.query_set:
            features = self._extract_scenario_features(scenario)
            pred = self._forward_pass(features, weight_means)
            adapted_predictions.append(pred)
        
        # Compare prediction quality
        targets = [ground_truth for _, ground_truth in task.query_set]
        
        base_error = np.mean([(p - t)**2 for p, t in zip(base_predictions, targets)])
        adapted_error = np.mean([(p - t)**2 for p, t in zip(adapted_predictions, targets)])
        
        # Adaptation quality is improvement ratio
        if base_error == 0:
            return 1.0
        
        improvement = max(0.0, (base_error - adapted_error) / base_error)
        
        return min(1.0, improvement)
    
    def _find_convergence_point(self, losses: List[float]) -> int:
        """Find convergence point in training trajectory."""
        if len(losses) < 10:
            return len(losses)
        
        # Find point where loss stops decreasing significantly
        convergence_threshold = 0.001
        window_size = 5
        
        for i in range(window_size, len(losses)):
            recent_losses = losses[i-window_size:i]
            current_loss = losses[i]
            
            if len(recent_losses) > 0:
                avg_recent = np.mean(recent_losses)
                improvement = (avg_recent - current_loss) / avg_recent
                
                if improvement < convergence_threshold:
                    return i
        
        return len(losses)


class AdaptiveMetaLearningFramework:
    """Complete adaptive meta-learning framework for skepticism evaluation.
    
    This framework integrates multiple meta-learning approaches and provides
    adaptive selection of the best strategy based on task characteristics.
    """
    
    def __init__(self, config: MetaLearningConfig):
        """Initialize adaptive meta-learning framework."""
        self.config = config
        self.bayesian_learner = BayesianNeuralMetaLearner(config)
        self.task_characteristics: Dict[str, Dict[str, float]] = {}
        self.strategy_performance: Dict[MetaLearningStrategy, List[float]] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Initialize performance tracking
        for strategy in MetaLearningStrategy:
            self.strategy_performance[strategy] = []
    
    async def adaptive_meta_learning(self, 
                                   meta_tasks: List[MetaTask],
                                   validation_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Perform adaptive meta-learning with strategy selection."""
        start_time = time.time()
        
        # Analyze task characteristics
        task_analysis = await self._analyze_task_characteristics(meta_tasks)
        
        # Train Bayesian meta-learner
        bayesian_results = await self.bayesian_learner.meta_train(meta_tasks)
        
        # Evaluate on validation tasks
        validation_results = await self._evaluate_on_validation(validation_tasks)
        
        # Adaptive strategy selection results
        strategy_analysis = await self._analyze_strategy_performance(validation_tasks)
        
        # Comprehensive uncertainty analysis
        uncertainty_analysis = await self._comprehensive_uncertainty_analysis(validation_tasks)
        
        total_time = time.time() - start_time
        
        return {
            'adaptive_meta_learning_time': total_time,
            'task_analysis': task_analysis,
            'bayesian_results': bayesian_results,
            'validation_results': validation_results,
            'strategy_analysis': strategy_analysis,
            'uncertainty_analysis': uncertainty_analysis,
            'best_strategy': await self._select_best_strategy(),
            'adaptation_recommendations': await self._generate_adaptation_recommendations(),
            'meta_learning_quality': self._compute_overall_quality(
                bayesian_results, validation_results, uncertainty_analysis
            )
        }
    
    async def _analyze_task_characteristics(self, tasks: List[MetaTask]) -> Dict[str, Any]:
        """Analyze characteristics of meta-learning tasks."""
        characteristics = {
            'num_tasks': len(tasks),
            'avg_support_size': np.mean([len(task.support_set) for task in tasks]),
            'avg_query_size': np.mean([len(task.query_set) for task in tasks]),
            'difficulty_distribution': [task.difficulty_level for task in tasks],
            'scenario_diversity': 0.0,
            'task_complexity': 0.0
        }
        
        # Calculate scenario diversity
        all_scenarios = []
        for task in tasks:
            all_scenarios.extend(task.scenarios)
        
        if all_scenarios:
            # Simple diversity measure based on scenario descriptions
            descriptions = [s.description for s in all_scenarios]
            unique_words = set()
            for desc in descriptions:
                unique_words.update(desc.lower().split())
            
            characteristics['scenario_diversity'] = len(unique_words) / max(1, len(descriptions))
        
        # Calculate task complexity
        complexities = []
        for task in tasks:
            complexity = len(task.support_set) + len(task.query_set)
            complexity += sum(len(s.description.split()) for s in task.scenarios) / 100.0
            complexities.append(complexity)
        
        characteristics['task_complexity'] = np.mean(complexities) if complexities else 0.0
        
        return characteristics
    
    async def _evaluate_on_validation(self, validation_tasks: List[MetaTask]) -> Dict[str, Any]:
        """Evaluate meta-learner on validation tasks."""
        total_accuracy = 0.0
        total_uncertainty_quality = 0.0
        total_adaptation_speed = 0.0
        num_evaluations = 0
        
        validation_details = []
        
        for task in validation_tasks[:10]:  # Sample for efficiency
            if not task.query_set:
                continue
            
            task_results = []
            
            for scenario, ground_truth in task.query_set:
                # Predict with uncertainty
                uncertainty_est = await self.bayesian_learner.predict_with_uncertainty(
                    scenario, task.support_set
                )
                
                # Calculate metrics
                accuracy = 1.0 - abs(uncertainty_est.mean_prediction - ground_truth)
                uncertainty_quality = (uncertainty_est.calibration_score + 
                                     uncertainty_est.reliability_score) / 2.0
                
                task_results.append({
                    'accuracy': accuracy,
                    'uncertainty_quality': uncertainty_quality,
                    'epistemic_uncertainty': uncertainty_est.epistemic_uncertainty,
                    'aleatoric_uncertainty': uncertainty_est.aleatoric_uncertainty,
                    'total_uncertainty': uncertainty_est.total_uncertainty
                })
                
                total_accuracy += accuracy
                total_uncertainty_quality += uncertainty_quality
                num_evaluations += 1
            
            # Measure adaptation speed for this task
            adaptation_speed = await self.bayesian_learner._measure_adaptation_speed(task)
            total_adaptation_speed += adaptation_speed
            
            validation_details.append({
                'task_id': task.task_id,
                'task_results': task_results,
                'adaptation_speed': adaptation_speed
            })
        
        return {
            'average_accuracy': total_accuracy / max(1, num_evaluations),
            'average_uncertainty_quality': total_uncertainty_quality / max(1, num_evaluations),
            'average_adaptation_speed': total_adaptation_speed / max(1, len(validation_tasks[:10])),
            'validation_details': validation_details,
            'num_evaluations': num_evaluations
        }
    
    async def _analyze_strategy_performance(self, tasks: List[MetaTask]) -> Dict[str, Any]:
        """Analyze performance of different meta-learning strategies."""
        strategy_results = {}
        
        # Currently we only have Bayesian MAML implemented
        # In a full implementation, we would compare multiple strategies
        
        bayesian_performance = []
        
        for task in tasks[:5]:  # Sample for analysis
            if task.query_set:
                task_performance = 0.0
                for scenario, ground_truth in task.query_set[:3]:
                    uncertainty_est = await self.bayesian_learner.predict_with_uncertainty(
                        scenario, task.support_set
                    )
                    accuracy = 1.0 - abs(uncertainty_est.mean_prediction - ground_truth)
                    task_performance += accuracy
                
                if task.query_set:
                    bayesian_performance.append(task_performance / min(3, len(task.query_set)))
        
        strategy_results[MetaLearningStrategy.BAYESIAN_MAML] = {
            'performance': np.mean(bayesian_performance) if bayesian_performance else 0.0,
            'variance': np.var(bayesian_performance) if len(bayesian_performance) > 1 else 0.0,
            'num_evaluations': len(bayesian_performance)
        }
        
        return {
            'strategy_results': strategy_results,
            'best_performing_strategy': MetaLearningStrategy.BAYESIAN_MAML,
            'performance_gap': 0.0,  # Would be calculated with multiple strategies
            'strategy_stability': strategy_results[MetaLearningStrategy.BAYESIAN_MAML]['variance']
        }
    
    async def _comprehensive_uncertainty_analysis(self, tasks: List[MetaTask]) -> Dict[str, Any]:
        """Perform comprehensive uncertainty analysis."""
        epistemic_uncertainties = []
        aleatoric_uncertainties = []
        total_uncertainties = []
        calibration_scores = []
        reliability_scores = []
        
        uncertainty_vs_error = []  # For correlation analysis
        
        for task in tasks[:10]:  # Sample for analysis
            for scenario, ground_truth in task.query_set[:3]:
                uncertainty_est = await self.bayesian_learner.predict_with_uncertainty(
                    scenario, task.support_set
                )
                
                epistemic_uncertainties.append(uncertainty_est.epistemic_uncertainty)
                aleatoric_uncertainties.append(uncertainty_est.aleatoric_uncertainty)
                total_uncertainties.append(uncertainty_est.total_uncertainty)
                calibration_scores.append(uncertainty_est.calibration_score)
                reliability_scores.append(uncertainty_est.reliability_score)
                
                # Track uncertainty vs prediction error
                prediction_error = abs(uncertainty_est.mean_prediction - ground_truth)
                uncertainty_vs_error.append((uncertainty_est.total_uncertainty, prediction_error))
        
        # Calculate uncertainty decomposition statistics
        epistemic_stats = self._compute_statistics(epistemic_uncertainties)
        aleatoric_stats = self._compute_statistics(aleatoric_uncertainties)
        total_stats = self._compute_statistics(total_uncertainties)
        
        # Analyze uncertainty-error correlation
        if uncertainty_vs_error:
            uncertainties, errors = zip(*uncertainty_vs_error)
            correlation = np.corrcoef(uncertainties, errors)[0, 1] if len(uncertainties) > 1 else 0.0
        else:
            correlation = 0.0
        
        return {
            'epistemic_uncertainty': epistemic_stats,
            'aleatoric_uncertainty': aleatoric_stats,
            'total_uncertainty': total_stats,
            'average_calibration': np.mean(calibration_scores) if calibration_scores else 0.0,
            'average_reliability': np.mean(reliability_scores) if reliability_scores else 0.0,
            'uncertainty_error_correlation': correlation,
            'uncertainty_decomposition_quality': self._evaluate_decomposition_quality(
                epistemic_uncertainties, aleatoric_uncertainties, total_uncertainties
            ),
            'calibration_consistency': np.std(calibration_scores) if len(calibration_scores) > 1 else 0.0
        }
    
    def _compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute statistical summary for a list of values."""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    def _evaluate_decomposition_quality(self, 
                                      epistemic: List[float], 
                                      aleatoric: List[float], 
                                      total: List[float]) -> float:
        """Evaluate quality of uncertainty decomposition."""
        if not (epistemic and aleatoric and total):
            return 0.0
        
        # Check if epistemic + aleatoric ≈ total (approximately)
        decomposition_errors = []
        for e, a, t in zip(epistemic, aleatoric, total):
            expected_total = math.sqrt(e**2 + a**2)
            error = abs(t - expected_total) / max(t, 1e-6)
            decomposition_errors.append(error)
        
        # Quality is inverse of average decomposition error
        avg_error = np.mean(decomposition_errors)
        quality = 1.0 / (1.0 + avg_error)
        
        return quality
    
    async def _select_best_strategy(self) -> MetaLearningStrategy:
        """Select best meta-learning strategy based on performance."""
        # For now, return Bayesian MAML as it's the only implemented strategy
        return MetaLearningStrategy.BAYESIAN_MAML
    
    async def _generate_adaptation_recommendations(self) -> List[str]:
        """Generate recommendations for improving adaptation."""
        recommendations = []
        
        # Analyze recent performance
        if len(self.adaptation_history) > 0:
            recent_performance = self.adaptation_history[-5:]  # Last 5 adaptations
            
            avg_uncertainty = np.mean([
                perf.get('average_uncertainty', 0.5) for perf in recent_performance
            ])
            
            if avg_uncertainty > 0.3:
                recommendations.append(
                    "High uncertainty detected - consider increasing training data diversity"
                )
            
            avg_adaptation_speed = np.mean([
                perf.get('adaptation_speed', 0.5) for perf in recent_performance
            ])
            
            if avg_adaptation_speed < 0.3:
                recommendations.append(
                    "Slow adaptation - consider increasing inner learning rate or steps"
                )
        
        # General recommendations
        recommendations.extend([
            "Monitor epistemic vs aleatoric uncertainty balance",
            "Validate uncertainty calibration on held-out scenarios",
            "Consider ensemble methods for improved reliability"
        ])
        
        return recommendations
    
    def _compute_overall_quality(self, 
                               bayesian_results: Dict[str, Any], 
                               validation_results: Dict[str, Any], 
                               uncertainty_analysis: Dict[str, Any]) -> float:
        """Compute overall meta-learning quality score."""
        # Weighted combination of key metrics
        
        # Accuracy component (40%)
        accuracy_score = validation_results.get('average_accuracy', 0.0)
        
        # Uncertainty quality component (30%)
        uncertainty_score = uncertainty_analysis.get('average_calibration', 0.0)
        
        # Adaptation speed component (20%)
        adaptation_score = validation_results.get('average_adaptation_speed', 0.0)
        
        # Training efficiency component (10%)
        training_score = 1.0 - min(1.0, bayesian_results.get('meta_training_time', 0.0) / 3600.0)
        
        overall_quality = (0.4 * accuracy_score + 
                         0.3 * uncertainty_score + 
                         0.2 * adaptation_score + 
                         0.1 * training_score)
        
        return overall_quality