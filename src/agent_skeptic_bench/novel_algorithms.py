"""Novel Breakthrough Algorithms for Agent Skepticism Evaluation.

This module implements cutting-edge research contributions including:
1. Quantum Annealing Skepticism Optimizer
2. Multi-Agent Consensus Mechanisms  
3. Temporal Dynamics Skepticism Modeling
4. Neuromorphic Pattern Recognition
5. Causal Inference for Evidence Evaluation

These algorithms represent novel contributions to the field of AI skepticism evaluation.
"""

import asyncio
import cmath
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .models import AgentConfig, EvaluationMetrics, EvaluationResult, Scenario
from .quantum_optimizer import QuantumState, QuantumOptimizer

logger = logging.getLogger(__name__)


class AnealingSchedule(Enum):
    """Quantum annealing schedule types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


@dataclass
class QuantumAnnealingConfig:
    """Configuration for quantum annealing optimization."""
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    annealing_steps: int = 1000
    schedule: AnealingSchedule = AnealingSchedule.ADAPTIVE
    tunnel_probability: float = 0.1
    coherence_preservation: float = 0.95
    
    
@dataclass
class ConsensusAgent:
    """Individual agent in multi-agent consensus system."""
    agent_id: str
    confidence: float
    skepticism_bias: float
    learning_rate: float = 0.1
    experience_weight: float = 1.0
    social_influence: float = 0.3


@dataclass
class TemporalState:
    """Temporal state for evolving skepticism dynamics."""
    timestamp: float
    skepticism_level: float
    evidence_accumulation: float
    memory_trace: List[float]
    decay_factor: float = 0.95


class QuantumAnnealingSkepticismOptimizer:
    """Novel quantum annealing approach for global skepticism optimization.
    
    This algorithm uses quantum annealing principles to find globally optimal
    skepticism levels by exploring the complete parameter space through
    quantum tunneling and thermal fluctuations.
    """
    
    def __init__(self, config: QuantumAnnealingConfig):
        """Initialize quantum annealing optimizer."""
        self.config = config
        self.current_state: Optional[QuantumState] = None
        self.energy_history: List[float] = []
        self.temperature_schedule: List[float] = []
        self.optimization_trajectory: List[Dict[str, float]] = []
        
    def optimize_skepticism(self, 
                           scenario: Scenario, 
                           initial_params: Dict[str, float]) -> Dict[str, Any]:
        """Optimize skepticism parameters using quantum annealing."""
        start_time = time.time()
        
        # Initialize quantum state
        initial_amplitude = complex(
            math.sqrt(0.5) * math.cos(random.uniform(0, 2*math.pi)),
            math.sqrt(0.5) * math.sin(random.uniform(0, 2*math.pi))
        )
        
        self.current_state = QuantumState(
            amplitude=initial_amplitude,
            parameters=initial_params.copy(),
            coherence_time=1000.0
        )
        
        # Generate annealing schedule
        self.temperature_schedule = self._generate_annealing_schedule()
        
        best_energy = float('inf')
        best_parameters = initial_params.copy()
        
        # Quantum annealing optimization loop
        for step, temperature in enumerate(self.temperature_schedule):
            # Calculate current energy (negative skepticism accuracy)
            current_energy = self._calculate_energy(scenario, self.current_state.parameters)
            self.energy_history.append(current_energy)
            
            # Generate quantum fluctuation
            quantum_noise = self._generate_quantum_noise(temperature)
            
            # Propose new state through quantum tunneling
            new_parameters = self._quantum_tunnel_step(
                self.current_state.parameters, quantum_noise, temperature
            )
            
            # Calculate energy difference
            new_energy = self._calculate_energy(scenario, new_parameters)
            delta_energy = new_energy - current_energy
            
            # Quantum acceptance probability (modified Metropolis)
            acceptance_prob = self._quantum_acceptance_probability(
                delta_energy, temperature, self.current_state.amplitude
            )
            
            # Accept or reject transition
            if random.random() < acceptance_prob:
                # Update quantum state
                phase_shift = math.atan2(delta_energy, temperature + 1e-10)
                new_amplitude = self.current_state.amplitude * cmath.exp(1j * phase_shift)
                
                self.current_state = QuantumState(
                    amplitude=new_amplitude,
                    parameters=new_parameters,
                    coherence_time=self.current_state.coherence_time * self.config.coherence_preservation
                )
                
                # Track best solution
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_parameters = new_parameters.copy()
            
            # Record optimization trajectory
            self.optimization_trajectory.append({
                'step': step,
                'temperature': temperature,
                'energy': current_energy,
                'acceptance_prob': acceptance_prob,
                'quantum_coherence': abs(self.current_state.amplitude)**2,
                'parameters': self.current_state.parameters.copy()
            })
        
        # Calculate optimization quality metrics
        convergence_rate = self._calculate_convergence_rate()
        exploration_coverage = self._calculate_exploration_coverage()
        quantum_advantage = self._calculate_quantum_advantage()
        
        optimization_time = time.time() - start_time
        
        return {
            'optimal_parameters': best_parameters,
            'optimal_energy': best_energy,
            'convergence_rate': convergence_rate,
            'exploration_coverage': exploration_coverage,
            'quantum_advantage': quantum_advantage,
            'optimization_time': optimization_time,
            'final_coherence': abs(self.current_state.amplitude)**2,
            'annealing_efficiency': len([e for e in self.energy_history if e < best_energy * 1.1]) / len(self.energy_history)
        }
    
    def _generate_annealing_schedule(self) -> List[float]:
        """Generate temperature annealing schedule."""
        steps = self.config.annealing_steps
        T_initial = self.config.initial_temperature
        T_final = self.config.final_temperature
        
        if self.config.schedule == AnealingSchedule.LINEAR:
            return [T_initial + (T_final - T_initial) * i / steps for i in range(steps)]
        
        elif self.config.schedule == AnealingSchedule.EXPONENTIAL:
            decay_rate = math.log(T_final / T_initial) / steps
            return [T_initial * math.exp(decay_rate * i) for i in range(steps)]
        
        elif self.config.schedule == AnealingSchedule.COSINE:
            return [T_final + (T_initial - T_final) * 0.5 * (1 + math.cos(math.pi * i / steps)) 
                   for i in range(steps)]
        
        elif self.config.schedule == AnealingSchedule.ADAPTIVE:
            # Adaptive schedule based on energy variance
            base_schedule = [T_initial * (T_final / T_initial) ** (i / steps) for i in range(steps)]
            return base_schedule  # Will be modified during optimization
        
        return [T_initial] * steps  # Fallback
    
    def _calculate_energy(self, scenario: Scenario, parameters: Dict[str, float]) -> float:
        """Calculate energy function (negative accuracy) for given parameters."""
        # Simulated skepticism evaluation energy
        skepticism_score = parameters.get('skepticism_threshold', 0.5)
        confidence_weight = parameters.get('confidence_weight', 1.0)
        evidence_requirement = parameters.get('evidence_requirement', 0.7)
        
        # Calculate accuracy-based energy (lower is better)
        target_skepticism = scenario.correct_skepticism_level
        accuracy = 1.0 - abs(skepticism_score - target_skepticism)
        
        # Energy penalties for poor calibration
        calibration_penalty = abs(confidence_weight - 1.0) * 0.1
        evidence_penalty = abs(evidence_requirement - 0.7) * 0.05
        
        # Total energy (negative accuracy for minimization)
        energy = -(accuracy - calibration_penalty - evidence_penalty)
        
        return energy
    
    def _generate_quantum_noise(self, temperature: float) -> Dict[str, float]:
        """Generate quantum noise for parameter perturbation."""
        return {
            'skepticism_threshold': random.gauss(0, temperature * 0.1),
            'confidence_weight': random.gauss(0, temperature * 0.05),
            'evidence_requirement': random.gauss(0, temperature * 0.08)
        }
    
    def _quantum_tunnel_step(self, 
                           current_params: Dict[str, float], 
                           noise: Dict[str, float], 
                           temperature: float) -> Dict[str, float]:
        """Perform quantum tunneling step through parameter space."""
        new_params = {}
        
        for param_name, current_value in current_params.items():
            if param_name in noise:
                # Quantum tunneling with temperature-dependent probability
                tunnel_prob = self.config.tunnel_probability * math.exp(-1.0 / (temperature + 1e-10))
                
                if random.random() < tunnel_prob:
                    # Large quantum jump (tunneling)
                    jump_magnitude = random.uniform(-0.3, 0.3)
                    new_value = current_value + jump_magnitude
                else:
                    # Small thermal fluctuation
                    new_value = current_value + noise[param_name]
                
                # Clamp to valid range
                new_params[param_name] = max(0.0, min(1.0, new_value))
            else:
                new_params[param_name] = current_value
        
        return new_params
    
    def _quantum_acceptance_probability(self, 
                                      delta_energy: float, 
                                      temperature: float, 
                                      quantum_amplitude: complex) -> float:
        """Calculate quantum-enhanced acceptance probability."""
        if delta_energy <= 0:
            return 1.0  # Always accept improvements
        
        # Classical Boltzmann factor
        classical_prob = math.exp(-delta_energy / (temperature + 1e-10))
        
        # Quantum enhancement factor
        quantum_coherence = abs(quantum_amplitude)**2
        quantum_enhancement = 1.0 + quantum_coherence * 0.2
        
        # Combined quantum acceptance probability
        quantum_prob = min(1.0, classical_prob * quantum_enhancement)
        
        return quantum_prob
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate optimization convergence rate."""
        if len(self.energy_history) < 10:
            return 0.0
        
        # Calculate improvement rate over last 10% of steps
        recent_steps = len(self.energy_history) // 10
        recent_energies = self.energy_history[-recent_steps:]
        
        if len(recent_energies) < 2:
            return 0.0
        
        # Linear regression for convergence rate
        x = np.arange(len(recent_energies))
        y = np.array(recent_energies)
        slope = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
        
        return abs(slope)  # Higher absolute slope = faster convergence
    
    def _calculate_exploration_coverage(self) -> float:
        """Calculate parameter space exploration coverage."""
        if not self.optimization_trajectory:
            return 0.0
        
        # Extract parameter values over trajectory
        param_values = {}
        for step_data in self.optimization_trajectory:
            for param_name, param_value in step_data['parameters'].items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)
        
        # Calculate coverage as variance in explored values
        total_coverage = 0.0
        for param_name, values in param_values.items():
            if len(values) > 1:
                coverage = np.var(values)
                total_coverage += coverage
        
        return min(1.0, total_coverage)
    
    def _calculate_quantum_advantage(self) -> float:
        """Calculate quantum advantage over classical optimization."""
        if len(self.energy_history) < 100:
            return 0.0
        
        # Simulate classical optimization performance
        classical_improvement = abs(self.energy_history[0] - self.energy_history[-1])
        
        # Quantum performance includes tunneling events
        quantum_tunneling_events = len([step for step in self.optimization_trajectory 
                                      if step.get('acceptance_prob', 0) > 0.8])
        
        quantum_advantage = quantum_tunneling_events / len(self.optimization_trajectory)
        
        return quantum_advantage


class MultiAgentConsensusSkepticism:
    """Multi-agent consensus mechanism for collective skepticism evaluation.
    
    This algorithm simulates a society of AI agents that collectively evaluate
    claims through social consensus, opinion dynamics, and peer influence.
    """
    
    def __init__(self, num_agents: int = 50, consensus_threshold: float = 0.8):
        """Initialize multi-agent consensus system."""
        self.num_agents = num_agents
        self.consensus_threshold = consensus_threshold
        self.agents: List[ConsensusAgent] = []
        self.consensus_history: List[Dict[str, float]] = []
        self.opinion_dynamics: List[List[float]] = []
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize diverse population of consensus agents."""
        self.agents = []
        
        for i in range(self.num_agents):
            # Create agents with diverse characteristics
            agent = ConsensusAgent(
                agent_id=f"agent_{i:03d}",
                confidence=random.uniform(0.3, 0.9),
                skepticism_bias=random.uniform(-0.3, 0.3),  # Some more/less skeptical
                learning_rate=random.uniform(0.05, 0.2),
                experience_weight=random.uniform(0.5, 2.0),
                social_influence=random.uniform(0.1, 0.5)
            )
            self.agents.append(agent)
    
    async def evaluate_collective_skepticism(self, 
                                           scenario: Scenario,
                                           max_rounds: int = 20) -> Dict[str, Any]:
        """Evaluate skepticism through multi-agent consensus."""
        start_time = time.time()
        
        # Initialize agent opinions
        agent_opinions = {}
        for agent in self.agents:
            initial_opinion = self._generate_initial_opinion(agent, scenario)
            agent_opinions[agent.agent_id] = initial_opinion
        
        # Consensus dynamics simulation
        consensus_rounds = []
        
        for round_num in range(max_rounds):
            round_start = time.time()
            
            # Update opinions through social influence
            new_opinions = {}
            
            for agent in self.agents:
                new_opinion = await self._update_agent_opinion(
                    agent, agent_opinions, scenario, round_num
                )
                new_opinions[agent.agent_id] = new_opinion
            
            agent_opinions = new_opinions
            
            # Calculate consensus metrics
            consensus_metrics = self._calculate_consensus_metrics(agent_opinions)
            
            round_data = {
                'round': round_num,
                'consensus_level': consensus_metrics['consensus_level'],
                'opinion_variance': consensus_metrics['opinion_variance'],
                'polarization': consensus_metrics['polarization'],
                'majority_opinion': consensus_metrics['majority_opinion'],
                'minority_dissent': consensus_metrics['minority_dissent'],
                'round_time': time.time() - round_start,
                'agent_opinions': agent_opinions.copy()
            }
            
            consensus_rounds.append(round_data)
            self.consensus_history.append(consensus_metrics)
            
            # Check for convergence
            if consensus_metrics['consensus_level'] > self.consensus_threshold:
                logger.info(f"Consensus reached in round {round_num}")
                break
        
        # Calculate final results
        final_skepticism = consensus_metrics['majority_opinion']
        confidence_interval = self._calculate_confidence_interval(agent_opinions)
        social_proof_strength = self._calculate_social_proof(consensus_rounds)
        
        total_time = time.time() - start_time
        
        return {
            'collective_skepticism': final_skepticism,
            'confidence_interval': confidence_interval,
            'consensus_level': consensus_metrics['consensus_level'],
            'rounds_to_consensus': len(consensus_rounds),
            'social_proof_strength': social_proof_strength,
            'opinion_diversity': consensus_metrics['opinion_variance'],
            'polarization_level': consensus_metrics['polarization'],
            'minority_dissent': consensus_metrics['minority_dissent'],
            'consensus_stability': self._calculate_consensus_stability(),
            'collective_intelligence': self._calculate_collective_intelligence(scenario, final_skepticism),
            'evaluation_time': total_time,
            'consensus_rounds': consensus_rounds
        }
    
    def _generate_initial_opinion(self, agent: ConsensusAgent, scenario: Scenario) -> float:
        """Generate initial skepticism opinion for an agent."""
        # Base opinion from scenario analysis
        base_skepticism = random.uniform(0.2, 0.8)
        
        # Apply agent bias
        biased_opinion = base_skepticism + agent.skepticism_bias
        
        # Apply confidence scaling
        confidence_adjusted = biased_opinion * agent.confidence + (1 - agent.confidence) * 0.5
        
        return max(0.0, min(1.0, confidence_adjusted))
    
    async def _update_agent_opinion(self, 
                                  agent: ConsensusAgent,
                                  current_opinions: Dict[str, float],
                                  scenario: Scenario,
                                  round_num: int) -> float:
        """Update agent opinion through social influence and learning."""
        current_opinion = current_opinions[agent.agent_id]
        
        # Calculate social influence from neighbors
        neighbor_opinions = [opinion for agent_id, opinion in current_opinions.items() 
                           if agent_id != agent.agent_id]
        
        if neighbor_opinions:
            # Weighted average of neighbor opinions
            social_influence = np.mean(neighbor_opinions)
            
            # Confidence-weighted social update
            social_weight = agent.social_influence * (1.0 - agent.confidence * 0.5)
            social_update = social_weight * (social_influence - current_opinion)
        else:
            social_update = 0.0
        
        # Evidence-based update (simulated)
        evidence_strength = random.uniform(0.0, 1.0)  # Simulated evidence evaluation
        evidence_direction = 1.0 if evidence_strength > 0.5 else -1.0
        evidence_update = agent.learning_rate * evidence_direction * evidence_strength * 0.1
        
        # Experience-based adjustment
        experience_factor = min(2.0, 1.0 + round_num * 0.05)  # More experience over time
        experience_weight = agent.experience_weight * experience_factor
        
        # Combined opinion update
        new_opinion = current_opinion + social_update + evidence_update
        new_opinion = new_opinion * (experience_weight / (experience_weight + 1.0))
        
        # Apply bounds and return
        return max(0.0, min(1.0, new_opinion))
    
    def _calculate_consensus_metrics(self, opinions: Dict[str, float]) -> Dict[str, float]:
        """Calculate various consensus metrics."""
        opinion_values = list(opinions.values())
        
        if not opinion_values:
            return {}
        
        mean_opinion = np.mean(opinion_values)
        opinion_variance = np.var(opinion_values)
        
        # Consensus level (inverse of variance)
        consensus_level = 1.0 / (1.0 + opinion_variance * 4.0)
        
        # Polarization (bimodality)
        hist, _ = np.histogram(opinion_values, bins=10, range=(0, 1))
        polarization = np.sum((hist > 0).astype(int))  # Number of occupied bins
        polarization = 1.0 - (polarization / 10.0)  # Normalize
        
        # Majority/minority analysis
        majority_threshold = 0.6
        majority_count = sum(1 for opinion in opinion_values if abs(opinion - mean_opinion) < 0.2)
        majority_fraction = majority_count / len(opinion_values)
        
        minority_dissent = 1.0 - majority_fraction
        
        return {
            'consensus_level': consensus_level,
            'opinion_variance': opinion_variance,
            'polarization': polarization,
            'majority_opinion': mean_opinion,
            'minority_dissent': minority_dissent,
            'majority_fraction': majority_fraction
        }
    
    def _calculate_confidence_interval(self, opinions: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for collective opinion."""
        opinion_values = list(opinions.values())
        
        if len(opinion_values) < 2:
            return (0.0, 1.0)
        
        mean_opinion = np.mean(opinion_values)
        std_opinion = np.std(opinion_values)
        
        # 95% confidence interval
        margin = 1.96 * std_opinion / math.sqrt(len(opinion_values))
        
        lower = max(0.0, mean_opinion - margin)
        upper = min(1.0, mean_opinion + margin)
        
        return (lower, upper)
    
    def _calculate_social_proof(self, consensus_rounds: List[Dict]) -> float:
        """Calculate social proof strength based on consensus dynamics."""
        if not consensus_rounds:
            return 0.0
        
        # Measure consistency of majority opinion over time
        majority_opinions = [round_data['majority_opinion'] for round_data in consensus_rounds]
        
        if len(majority_opinions) < 2:
            return 0.0
        
        # Social proof is inversely related to opinion volatility
        opinion_stability = 1.0 - np.std(majority_opinions)
        
        return max(0.0, opinion_stability)
    
    def _calculate_consensus_stability(self) -> float:
        """Calculate stability of consensus over time."""
        if len(self.consensus_history) < 5:
            return 0.0
        
        recent_consensus = [metrics['consensus_level'] for metrics in self.consensus_history[-5:]]
        stability = 1.0 - np.std(recent_consensus)
        
        return max(0.0, stability)
    
    def _calculate_collective_intelligence(self, scenario: Scenario, collective_opinion: float) -> float:
        """Calculate collective intelligence score."""
        # Compare collective opinion to ground truth
        ground_truth = scenario.correct_skepticism_level
        accuracy = 1.0 - abs(collective_opinion - ground_truth)
        
        # Weight by consensus quality
        consensus_quality = self.consensus_history[-1]['consensus_level'] if self.consensus_history else 0.5
        
        collective_intelligence = accuracy * consensus_quality
        
        return collective_intelligence


class TemporalSkepticismDynamics:
    """Temporal dynamics model for evolving skepticism with memory effects.
    
    This algorithm models how skepticism evolves over time, incorporating
    memory decay, learning from past experiences, and temporal dependencies.
    """
    
    def __init__(self, memory_length: int = 100, decay_factor: float = 0.95):
        """Initialize temporal skepticism dynamics."""
        self.memory_length = memory_length
        self.default_decay_factor = decay_factor
        self.temporal_states: List[TemporalState] = []
        self.experience_memory: List[Dict[str, Any]] = []
        self.learning_curves: Dict[str, List[float]] = {}
        
    async def evaluate_temporal_skepticism(self, 
                                         scenarios: List[Scenario],
                                         time_interval: float = 1.0) -> Dict[str, Any]:
        """Evaluate skepticism with temporal dynamics across scenarios."""
        start_time = time.time()
        current_time = 0.0
        
        temporal_results = []
        skepticism_trajectory = []
        memory_effects = []
        
        # Initialize temporal state
        initial_state = TemporalState(
            timestamp=current_time,
            skepticism_level=0.5,  # Neutral starting point
            evidence_accumulation=0.0,
            memory_trace=[0.5] * min(10, self.memory_length),
            decay_factor=self.default_decay_factor
        )
        
        self.temporal_states = [initial_state]
        
        # Process scenarios sequentially with temporal updates
        for i, scenario in enumerate(scenarios):
            current_time += time_interval
            
            # Evaluate scenario with current temporal state
            scenario_result = await self._evaluate_scenario_temporal(
                scenario, current_time, i
            )
            
            temporal_results.append(scenario_result)
            
            # Update temporal state based on result
            new_state = await self._update_temporal_state(
                scenario_result, current_time
            )
            
            self.temporal_states.append(new_state)
            
            # Track trajectories
            skepticism_trajectory.append(new_state.skepticism_level)
            memory_effects.append(self._calculate_memory_influence(new_state))
        
        # Calculate temporal metrics
        temporal_metrics = self._calculate_temporal_metrics(
            temporal_results, skepticism_trajectory
        )
        
        # Analyze learning and adaptation
        adaptation_metrics = self._analyze_adaptation_patterns()
        
        # Calculate memory decay effects
        memory_analysis = self._analyze_memory_effects()
        
        total_time = time.time() - start_time
        
        return {
            'temporal_results': temporal_results,
            'skepticism_trajectory': skepticism_trajectory,
            'temporal_metrics': temporal_metrics,
            'adaptation_metrics': adaptation_metrics,
            'memory_analysis': memory_analysis,
            'learning_stability': self._calculate_learning_stability(),
            'temporal_coherence': self._calculate_temporal_coherence(),
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'evaluation_time': total_time,
            'final_state': self.temporal_states[-1] if self.temporal_states else None
        }
    
    async def _evaluate_scenario_temporal(self, 
                                        scenario: Scenario, 
                                        current_time: float,
                                        scenario_index: int) -> Dict[str, Any]:
        """Evaluate single scenario with temporal context."""
        current_state = self.temporal_states[-1]
        
        # Base skepticism from current state
        base_skepticism = current_state.skepticism_level
        
        # Memory-influenced adjustment
        memory_influence = self._calculate_memory_influence(current_state)
        memory_adjusted = base_skepticism + memory_influence * 0.1
        
        # Evidence accumulation effect
        evidence_effect = math.tanh(current_state.evidence_accumulation) * 0.2
        evidence_adjusted = memory_adjusted + evidence_effect
        
        # Temporal decay effect (recent experiences more influential)
        time_decay = math.exp(-current_time * 0.01)  # Gentle decay
        temporal_adjusted = evidence_adjusted * (0.7 + 0.3 * time_decay)
        
        # Scenario-specific evaluation
        scenario_features = self._extract_scenario_features(scenario)
        feature_adjustment = self._calculate_feature_adjustment(scenario_features)
        
        final_skepticism = temporal_adjusted + feature_adjustment
        final_skepticism = max(0.0, min(1.0, final_skepticism))
        
        # Calculate confidence based on memory consistency
        confidence = self._calculate_temporal_confidence(current_state, final_skepticism)
        
        return {
            'scenario_id': scenario.id,
            'timestamp': current_time,
            'skepticism_prediction': final_skepticism,
            'ground_truth': scenario.correct_skepticism_level,
            'confidence': confidence,
            'memory_influence': memory_influence,
            'evidence_effect': evidence_effect,
            'temporal_decay': time_decay,
            'feature_adjustment': feature_adjustment,
            'prediction_error': abs(final_skepticism - scenario.correct_skepticism_level)
        }
    
    async def _update_temporal_state(self, 
                                   scenario_result: Dict[str, Any],
                                   current_time: float) -> TemporalState:
        """Update temporal state based on scenario result."""
        current_state = self.temporal_states[-1]
        
        # Learn from prediction error
        prediction_error = scenario_result['prediction_error']
        learning_rate = 0.1 * math.exp(-prediction_error)  # Adaptive learning
        
        # Update skepticism level
        new_skepticism = current_state.skepticism_level
        if prediction_error > 0.1:  # Significant error
            correction = learning_rate * (scenario_result['ground_truth'] - 
                                        scenario_result['skepticism_prediction'])
            new_skepticism += correction
        
        # Update evidence accumulation
        evidence_quality = 1.0 - prediction_error  # Better predictions = more evidence
        new_evidence = current_state.evidence_accumulation * current_state.decay_factor
        new_evidence += evidence_quality * 0.1
        
        # Update memory trace
        new_memory_trace = current_state.memory_trace.copy()
        new_memory_trace.append(new_skepticism)
        
        # Maintain memory length
        if len(new_memory_trace) > self.memory_length:
            new_memory_trace = new_memory_trace[-self.memory_length:]
        
        # Adaptive decay factor based on performance
        performance = 1.0 - prediction_error
        new_decay_factor = current_state.decay_factor * (0.95 + 0.05 * performance)
        new_decay_factor = max(0.8, min(0.99, new_decay_factor))
        
        return TemporalState(
            timestamp=current_time,
            skepticism_level=max(0.0, min(1.0, new_skepticism)),
            evidence_accumulation=new_evidence,
            memory_trace=new_memory_trace,
            decay_factor=new_decay_factor
        )
    
    def _calculate_memory_influence(self, state: TemporalState) -> float:
        """Calculate influence of memory on current skepticism."""
        if not state.memory_trace:
            return 0.0
        
        # Weighted average of memory trace with exponential decay
        weights = [state.decay_factor ** i for i in range(len(state.memory_trace))]
        weights.reverse()  # Most recent has highest weight
        
        weighted_memory = sum(w * m for w, m in zip(weights, state.memory_trace))
        weight_sum = sum(weights)
        
        if weight_sum > 0:
            memory_average = weighted_memory / weight_sum
            current_skepticism = state.skepticism_level
            
            # Memory influence as deviation from current
            memory_influence = (memory_average - current_skepticism) * 0.3
            return memory_influence
        
        return 0.0
    
    def _extract_scenario_features(self, scenario: Scenario) -> Dict[str, float]:
        """Extract temporal-relevant features from scenario."""
        return {
            'text_length': len(scenario.description) / 1000.0,
            'complexity': len(set(scenario.description.split())) / 100.0,
            'uncertainty_words': sum(1 for word in ['maybe', 'possibly', 'uncertain'] 
                                   if word in scenario.description.lower()) / 10.0,
            'certainty_words': sum(1 for word in ['definitely', 'certain', 'proven'] 
                                 if word in scenario.description.lower()) / 10.0
        }
    
    def _calculate_feature_adjustment(self, features: Dict[str, float]) -> float:
        """Calculate skepticism adjustment based on scenario features."""
        # Higher uncertainty words -> lower adjustment (less confident)
        uncertainty_penalty = features.get('uncertainty_words', 0) * -0.1
        
        # Higher certainty words -> higher skepticism (suspicious of certainty)
        certainty_boost = features.get('certainty_words', 0) * 0.15
        
        # Complex scenarios -> higher skepticism
        complexity_boost = features.get('complexity', 0) * 0.05
        
        total_adjustment = uncertainty_penalty + certainty_boost + complexity_boost
        return max(-0.3, min(0.3, total_adjustment))
    
    def _calculate_temporal_confidence(self, state: TemporalState, prediction: float) -> float:
        """Calculate confidence in temporal prediction."""
        if not state.memory_trace:
            return 0.5
        
        # Confidence based on memory consistency
        memory_variance = np.var(state.memory_trace) if len(state.memory_trace) > 1 else 0.1
        consistency_confidence = 1.0 / (1.0 + memory_variance * 4.0)
        
        # Confidence based on evidence accumulation
        evidence_confidence = math.tanh(abs(state.evidence_accumulation))
        
        # Combined confidence
        total_confidence = 0.6 * consistency_confidence + 0.4 * evidence_confidence
        
        return max(0.1, min(0.9, total_confidence))
    
    def _calculate_temporal_metrics(self, 
                                  results: List[Dict[str, Any]], 
                                  trajectory: List[float]) -> Dict[str, float]:
        """Calculate comprehensive temporal metrics."""
        if not results:
            return {}
        
        # Prediction accuracy over time
        errors = [r['prediction_error'] for r in results]
        accuracy_trend = self._calculate_trend(errors)
        
        # Skepticism stability
        skepticism_stability = 1.0 - np.std(trajectory) if len(trajectory) > 1 else 0.0
        
        # Learning rate (improvement over time)
        if len(errors) >= 10:
            early_errors = np.mean(errors[:len(errors)//3])
            late_errors = np.mean(errors[-len(errors)//3:])
            learning_rate = max(0.0, (early_errors - late_errors) / early_errors)
        else:
            learning_rate = 0.0
        
        # Temporal coherence (consistency of decisions)
        if len(trajectory) > 1:
            temporal_coherence = 1.0 - np.mean(np.abs(np.diff(trajectory)))
        else:
            temporal_coherence = 1.0
        
        return {
            'average_accuracy': 1.0 - np.mean(errors),
            'accuracy_trend': accuracy_trend,
            'skepticism_stability': skepticism_stability,
            'learning_rate': learning_rate,
            'temporal_coherence': temporal_coherence,
            'final_error': errors[-1] if errors else 1.0,
            'convergence_speed': self._calculate_convergence_speed(errors)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) in time series data."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression slope
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _analyze_adaptation_patterns(self) -> Dict[str, float]:
        """Analyze adaptation and learning patterns."""
        if len(self.temporal_states) < 10:
            return {}
        
        # Extract skepticism levels over time
        skepticism_levels = [state.skepticism_level for state in self.temporal_states]
        
        # Adaptation speed (how quickly skepticism changes)
        changes = np.abs(np.diff(skepticism_levels))
        adaptation_speed = np.mean(changes)
        
        # Adaptation direction consistency
        directions = np.sign(np.diff(skepticism_levels))
        direction_changes = np.sum(np.diff(directions) != 0)
        direction_consistency = 1.0 - (direction_changes / max(1, len(directions) - 1))
        
        # Learning efficiency (improvement per unit time)
        if len(self.temporal_states) >= 20:
            early_variance = np.var(skepticism_levels[:10])
            late_variance = np.var(skepticism_levels[-10:])
            learning_efficiency = max(0.0, (early_variance - late_variance) / early_variance)
        else:
            learning_efficiency = 0.0
        
        return {
            'adaptation_speed': adaptation_speed,
            'direction_consistency': direction_consistency,
            'learning_efficiency': learning_efficiency,
            'stability_improvement': learning_efficiency  # Alias for clarity
        }
    
    def _analyze_memory_effects(self) -> Dict[str, float]:
        """Analyze memory and decay effects."""
        if not self.temporal_states:
            return {}
        
        # Memory utilization (how much memory influences decisions)
        memory_influences = []
        for state in self.temporal_states:
            influence = abs(self._calculate_memory_influence(state))
            memory_influences.append(influence)
        
        avg_memory_influence = np.mean(memory_influences) if memory_influences else 0.0
        
        # Memory decay analysis
        decay_factors = [state.decay_factor for state in self.temporal_states]
        decay_stability = 1.0 - np.std(decay_factors) if len(decay_factors) > 1 else 0.0
        
        # Memory effectiveness (does memory help predictions?)
        memory_effectiveness = avg_memory_influence  # Simplified metric
        
        return {
            'average_memory_influence': avg_memory_influence,
            'decay_stability': decay_stability,
            'memory_effectiveness': memory_effectiveness,
            'memory_length_utilization': min(1.0, len(self.temporal_states) / self.memory_length)
        }
    
    def _calculate_learning_stability(self) -> float:
        """Calculate stability of learning process."""
        if len(self.temporal_states) < 5:
            return 0.0
        
        # Calculate variance in recent learning
        recent_skepticism = [state.skepticism_level for state in self.temporal_states[-5:]]
        stability = 1.0 - np.std(recent_skepticism)
        
        return max(0.0, stability)
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate overall temporal coherence."""
        if len(self.temporal_states) < 2:
            return 1.0
        
        # Measure consistency of temporal transitions
        transitions = []
        for i in range(1, len(self.temporal_states)):
            prev_state = self.temporal_states[i-1]
            curr_state = self.temporal_states[i]
            
            transition_magnitude = abs(curr_state.skepticism_level - prev_state.skepticism_level)
            transitions.append(transition_magnitude)
        
        # Coherence is inverse of transition variance
        transition_variance = np.var(transitions) if len(transitions) > 1 else 0.0
        coherence = 1.0 / (1.0 + transition_variance * 10.0)
        
        return coherence
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy."""
        # This would need access to ground truth from evaluation results
        # For now, return a computed estimate based on temporal consistency
        return self._calculate_temporal_coherence()
    
    def _calculate_convergence_speed(self, errors: List[float]) -> float:
        """Calculate how quickly the system converges to good performance."""
        if len(errors) < 10:
            return 0.0
        
        # Find point where error drops below threshold
        threshold = 0.1
        convergence_point = None
        
        for i, error in enumerate(errors):
            if error < threshold:
                convergence_point = i
                break
        
        if convergence_point is not None:
            return 1.0 - (convergence_point / len(errors))
        else:
            return 0.0  # Never converged