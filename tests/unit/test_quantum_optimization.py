"""Unit tests for quantum-inspired optimization features."""

import pytest
import math
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple

from src.agent_skeptic_bench.algorithms.optimization import (
    QuantumInspiredOptimizer, 
    SkepticismCalibrator,
    QuantumState
)
from src.agent_skeptic_bench.models import Scenario, SkepticResponse, EvaluationMetrics, ScenarioCategory


class TestQuantumState:
    """Test quantum state representation."""
    
    def test_quantum_state_creation(self):
        """Test quantum state initialization."""
        amplitude = complex(0.707, 0.707)
        parameters = {"temperature": 0.5, "threshold": 0.7}
        
        state = QuantumState(
            amplitude=amplitude,
            probability=0.0,  # Will be calculated
            parameters=parameters
        )
        
        assert state.amplitude == amplitude
        assert abs(state.probability - 0.5) < 0.1  # |amplitude|^2
        assert state.parameters == parameters
    
    def test_probability_calculation(self):
        """Test quantum probability calculation."""
        test_cases = [
            (complex(1, 0), 1.0),
            (complex(0.707, 0.707), 0.5),
            (complex(0, 1), 1.0),
            (complex(0.5, 0.866), 1.0)
        ]
        
        for amplitude, expected_prob in test_cases:
            state = QuantumState(
                amplitude=amplitude,
                probability=0.0,
                parameters={}
            )
            assert abs(state.probability - expected_prob) < 0.1


class TestQuantumInspiredOptimizer:
    """Test quantum-inspired optimization algorithm."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return QuantumInspiredOptimizer(
            population_size=10,
            max_generations=5,
            mutation_rate=0.1
        )
    
    @pytest.fixture
    def sample_bounds(self):
        """Sample parameter bounds."""
        return {
            "temperature": (0.1, 1.0),
            "threshold": (0.3, 0.8),
            "weight": (0.5, 1.5)
        }
    
    @pytest.fixture
    def sample_evaluation_data(self):
        """Sample evaluation data for optimization."""
        scenario = Scenario(
            id="test_scenario",
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title="Test Scenario",
            description="Test description",
            correct_skepticism_level=0.8,
            metadata={}
        )
        
        response = SkepticResponse(
            agent_id="test_agent",
            scenario_id="test_scenario",
            response_text="Test response",
            confidence_level=0.7,
            response_time_ms=500
        )
        
        metrics = EvaluationMetrics(
            skepticism_calibration=0.85,
            evidence_standard_score=0.75,
            red_flag_detection=0.80,
            reasoning_quality=0.90
        )
        
        return [(scenario, response, metrics)]
    
    def test_population_initialization(self, optimizer, sample_bounds):
        """Test quantum population initialization."""
        optimizer.initialize_population(sample_bounds)
        
        assert len(optimizer.population) == optimizer.population_size
        
        for state in optimizer.population:
            assert isinstance(state, QuantumState)
            assert len(state.parameters) == len(sample_bounds)
            
            # Check parameter bounds
            for param_name, (min_val, max_val) in sample_bounds.items():
                param_value = state.parameters[param_name]
                assert min_val <= param_value <= max_val
    
    def test_fitness_function(self, optimizer, sample_evaluation_data):
        """Test fitness function calculation."""
        parameters = {
            "temperature": 0.5,
            "threshold": 0.6,
            "weight": 1.0
        }
        
        fitness = optimizer.fitness_function(parameters, sample_evaluation_data)
        
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0
    
    def test_entanglement_bonus_calculation(self, optimizer):
        """Test quantum entanglement bonus calculation."""
        # Test with correlated parameters
        correlated_params = {"a": 0.5, "b": 0.5}
        bonus1 = optimizer._calculate_entanglement_bonus(correlated_params)
        
        # Test with uncorrelated parameters
        uncorrelated_params = {"a": 0.1, "b": 0.9}
        bonus2 = optimizer._calculate_entanglement_bonus(uncorrelated_params)
        
        assert isinstance(bonus1, float)
        assert isinstance(bonus2, float)
        assert 0.0 <= bonus1 <= 0.05
        assert 0.0 <= bonus2 <= 0.05
    
    def test_quantum_rotation(self, optimizer):
        """Test quantum rotation operator."""
        initial_state = QuantumState(
            amplitude=complex(0.707, 0.707),
            probability=0.5,
            parameters={"temp": 0.5}
        )
        
        rotated_state = optimizer.quantum_rotation(initial_state, 0.8)
        
        assert isinstance(rotated_state, QuantumState)
        assert rotated_state.amplitude != initial_state.amplitude
        assert "temp" in rotated_state.parameters
    
    def test_optimization_process(self, optimizer, sample_bounds, sample_evaluation_data):
        """Test complete optimization process."""
        optimal_params = optimizer.optimize(sample_bounds, sample_evaluation_data)
        
        assert isinstance(optimal_params, dict)
        assert len(optimal_params) == len(sample_bounds)
        
        # Check that parameters are within bounds
        for param_name, (min_val, max_val) in sample_bounds.items():
            param_value = optimal_params[param_name]
            assert min_val <= param_value <= max_val
        
        # Check that fitness history is recorded
        assert len(optimizer.fitness_history) == optimizer.max_generations
        assert all(isinstance(f, float) for f in optimizer.fitness_history)


class TestSkepticismCalibrator:
    """Test skepticism calibration system."""
    
    @pytest.fixture
    def calibrator(self):
        """Create calibrator instance."""
        return SkepticismCalibrator()
    
    @pytest.fixture
    def sample_historical_data(self):
        """Sample historical evaluation data."""
        data = []
        for i in range(5):
            scenario = Scenario(
                id=f"scenario_{i}",
                category=ScenarioCategory.FACTUAL_CLAIMS,
                title=f"Scenario {i}",
                description="Test description",
                correct_skepticism_level=0.7 + (i * 0.05),
                metadata={"difficulty": "medium"}
            )
            
            response = SkepticResponse(
                agent_id="test_agent",
                scenario_id=f"scenario_{i}",
                response_text="Test response",
                confidence_level=0.6 + (i * 0.05),
                response_time_ms=500 + (i * 50)
            )
            
            metrics = EvaluationMetrics(
                skepticism_calibration=0.8 + (i * 0.02),
                evidence_standard_score=0.75 + (i * 0.03),
                red_flag_detection=0.80 + (i * 0.01),
                reasoning_quality=0.85 + (i * 0.02)
            )
            
            data.append((scenario, response, metrics))
        
        return data
    
    def test_parameter_calibration(self, calibrator, sample_historical_data):
        """Test agent parameter calibration."""
        target_metrics = {
            "skepticism_calibration": 0.85,
            "evidence_standard_score": 0.80,
            "red_flag_detection": 0.85,
            "reasoning_quality": 0.90
        }
        
        optimal_params = calibrator.calibrate_agent_parameters(
            sample_historical_data, 
            target_metrics
        )
        
        assert isinstance(optimal_params, dict)
        assert "temperature" in optimal_params
        assert "skepticism_threshold" in optimal_params
        assert "evidence_weight" in optimal_params
        
        # Check parameter ranges
        assert 0.1 <= optimal_params["temperature"] <= 1.0
        assert 0.3 <= optimal_params["skepticism_threshold"] <= 0.8
        assert 0.5 <= optimal_params["evidence_weight"] <= 1.5
    
    def test_skepticism_prediction(self, calibrator):
        """Test optimal skepticism prediction."""
        scenario = Scenario(
            id="test_scenario",
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title="Test Scenario",
            description="A claim about climate change being false",
            correct_skepticism_level=0.9,
            metadata={"evidence_quality": 0.2, "plausibility": 0.1}
        )
        
        agent_params = {
            "skepticism_threshold": 0.6,
            "evidence_weight": 1.2,
            "temperature": 0.7
        }
        
        predicted_skepticism = calibrator.predict_optimal_skepticism(scenario, agent_params)
        
        assert isinstance(predicted_skepticism, float)
        assert 0.0 <= predicted_skepticism <= 1.0
    
    def test_quantum_uncertainty_calculation(self, calibrator):
        """Test quantum uncertainty calculation."""
        # Test with simple scenario
        simple_scenario = Scenario(
            id="simple",
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title="Simple Test",
            description="Short description",
            correct_skepticism_level=0.5,
            metadata={}
        )
        
        uncertainty1 = calibrator._calculate_quantum_uncertainty(simple_scenario)
        
        # Test with complex scenario
        complex_scenario = Scenario(
            id="complex",
            category=ScenarioCategory.FACTUAL_CLAIMS,
            title="Complex Test",
            description="This is a much longer and more complex description that should result in higher uncertainty calculations due to its increased complexity and length",
            correct_skepticism_level=0.8,
            metadata={"evidence_quality": 0.3, "plausibility": 0.2}
        )
        
        uncertainty2 = calibrator._calculate_quantum_uncertainty(complex_scenario)
        
        assert isinstance(uncertainty1, float)
        assert isinstance(uncertainty2, float)
        assert 0.0 <= uncertainty1 <= 1.0
        assert 0.0 <= uncertainty2 <= 1.0
    
    def test_calibration_report_generation(self, calibrator, sample_historical_data):
        """Test calibration report generation."""
        # Run calibration first
        calibrator.calibrate_agent_parameters(sample_historical_data)
        
        report = calibrator.get_calibration_report()
        
        assert isinstance(report, dict)
        assert "total_calibrations" in report
        assert "latest_calibration" in report
        assert "parameter_evolution" in report
        assert "optimization_performance" in report
        assert "recommendations" in report
        
        assert report["total_calibrations"] >= 1
        assert isinstance(report["recommendations"], list)
    
    def test_parameter_evolution_analysis(self, calibrator, sample_historical_data):
        """Test parameter evolution analysis."""
        # Run multiple calibrations
        for _ in range(3):
            calibrator.calibrate_agent_parameters(sample_historical_data)
        
        evolution = calibrator._analyze_parameter_evolution()
        
        assert isinstance(evolution, dict)
        assert len(evolution) >= 1  # Should have parameter evolution data
        
        for param_name, values in evolution.items():
            assert isinstance(values, list)
            assert len(values) == 3  # Three calibrations
    
    def test_optimization_performance_analysis(self, calibrator, sample_historical_data):
        """Test optimization performance analysis."""
        # Run calibration
        calibrator.calibrate_agent_parameters(sample_historical_data)
        
        performance = calibrator._analyze_optimization_performance()
        
        assert isinstance(performance, dict)
        if "average_final_fitness" in performance:
            assert isinstance(performance["average_final_fitness"], float)
        if "optimization_stability" in performance:
            assert isinstance(performance["optimization_stability"], float)
    
    def test_recommendation_generation(self, calibrator):
        """Test recommendation generation."""
        recommendations = calibrator._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1
        assert all(isinstance(rec, str) for rec in recommendations)


class TestQuantumValidation:
    """Test quantum validation features."""
    
    def test_quantum_coherence_validation(self):
        """Test quantum coherence validation."""
        from src.agent_skeptic_bench.validation import quantum_validator
        
        # Create sample evaluation results
        evaluation_results = []
        for i in range(3):
            scenario = Scenario(
                id=f"test_{i}",
                category=ScenarioCategory.FACTUAL_CLAIMS,
                title="Test",
                description="Test",
                correct_skepticism_level=0.8,
                metadata={}
            )
            
            response = SkepticResponse(
                agent_id="test",
                scenario_id=f"test_{i}",
                response_text="Test",
                confidence_level=0.75 + (i * 0.05),
                response_time_ms=500
            )
            
            metrics = EvaluationMetrics(
                skepticism_calibration=0.85,
                evidence_standard_score=0.80,
                red_flag_detection=0.75,
                reasoning_quality=0.90
            )
            
            evaluation_results.append((scenario, response, metrics))
        
        validation = quantum_validator.validate_quantum_coherence(evaluation_results)
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "quantum_coherence" in validation
        assert "phase_alignment" in validation
        assert isinstance(validation["valid"], bool)
        assert isinstance(validation["quantum_coherence"], float)
    
    def test_parameter_entanglement_validation(self):
        """Test parameter entanglement validation."""
        from src.agent_skeptic_bench.validation import quantum_validator
        
        parameters = {
            "temperature": 0.7,
            "threshold": 0.6,
            "weight": 1.2,
            "adjustment": 0.1
        }
        
        validation = quantum_validator.validate_parameter_entanglement(parameters)
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "entanglement_strength" in validation
        assert "entanglement_matrix" in validation
        assert isinstance(validation["valid"], bool)
        assert isinstance(validation["entanglement_strength"], float)
        assert isinstance(validation["entanglement_matrix"], list)
    
    def test_uncertainty_principle_validation(self):
        """Test quantum uncertainty principle validation."""
        from src.agent_skeptic_bench.validation import quantum_validator
        
        response = SkepticResponse(
            agent_id="test",
            scenario_id="test",
            response_text="Test response with reasonable length",
            confidence_level=0.65,
            evidence_requests=["peer review", "replication", "control group"],
            reasoning_steps=["analysis", "comparison", "conclusion"],
            response_time_ms=750
        )
        
        validation = quantum_validator.validate_uncertainty_principle(response, 0.7)
        
        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "measured_uncertainty" in validation
        assert "uncertainty_product" in validation
        assert "satisfies_uncertainty_principle" in validation
        assert isinstance(validation["valid"], bool)
        assert isinstance(validation["measured_uncertainty"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])