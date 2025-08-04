"""Unit tests for the SkepticBenchmark class."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch

from agent_skeptic_bench import SkepticBenchmark, Scenario, ScenarioCategory
from agent_skeptic_bench.agents import BaseSkepticAgent
from tests.conftest import create_test_scenarios


class TestSkepticBenchmark:
    """Test suite for SkepticBenchmark class."""
    
    def test_benchmark_initialization(self):
        """Test that SkepticBenchmark initializes correctly."""
        benchmark = SkepticBenchmark()
        assert benchmark is not None
        assert benchmark.scenario_loader is not None
        assert benchmark.agent_factory is not None
        assert benchmark.metrics_calculator is not None
        assert benchmark._active_sessions == {}
    
    def test_get_scenarios(self, benchmark_instance):
        """Test getting scenarios with filtering."""
        # Mock scenario loader
        mock_scenarios = create_test_scenarios(3)
        benchmark_instance.scenario_loader.load_scenarios = Mock(return_value=mock_scenarios)
        
        # Test getting all scenarios
        scenarios = benchmark_instance.get_scenarios()
        assert len(scenarios) == 3
        
        # Test with category filter
        scenarios = benchmark_instance.get_scenarios(categories=[ScenarioCategory.FACTUAL_CLAIMS])
        benchmark_instance.scenario_loader.load_scenarios.assert_called_with([ScenarioCategory.FACTUAL_CLAIMS])
        
        # Test with limit
        scenarios = benchmark_instance.get_scenarios(limit=2)
        assert len(scenarios) <= 2
    
    def test_get_scenario_by_id(self, benchmark_instance, sample_scenario):
        """Test getting a specific scenario by ID."""
        benchmark_instance.scenario_loader.get_scenario = Mock(return_value=sample_scenario)
        
        scenario = benchmark_instance.get_scenario("test_scenario_001")
        assert scenario == sample_scenario
        benchmark_instance.scenario_loader.get_scenario.assert_called_with("test_scenario_001")
    
    @pytest.mark.asyncio
    async def test_evaluate_scenario(self, benchmark_instance, sample_scenario, sample_agent_config):
        """Test evaluating a single scenario."""
        # Mock agent
        mock_agent = AsyncMock(spec=BaseSkepticAgent)
        mock_agent.config = sample_agent_config
        
        # Mock agent response
        from agent_skeptic_bench.models import SkepticResponse
        mock_response = SkepticResponse(
            agent_id="test_agent",
            scenario_id=sample_scenario.id,
            response_text="Test response",
            confidence_level=0.8,
            evidence_requests=["evidence"],
            red_flags_identified=["red_flag"],
            reasoning_steps=["reasoning"],
            response_time_ms=1000
        )
        mock_agent.evaluate_claim.return_value = mock_response
        
        # Mock metrics calculator
        from agent_skeptic_bench.models import EvaluationMetrics
        mock_metrics = EvaluationMetrics(
            skepticism_calibration=0.8,
            evidence_standard_score=0.7,
            red_flag_detection=0.9,
            reasoning_quality=0.8,
            overall_score=0.8
        )
        benchmark_instance.metrics_calculator.calculate_metrics = Mock(return_value=mock_metrics)
        
        # Test evaluation
        result = await benchmark_instance.evaluate_scenario(mock_agent, sample_scenario)
        
        assert result is not None
        assert result.scenario == sample_scenario
        assert result.response == mock_response
        assert result.metrics == mock_metrics
        assert result.passed_evaluation == True  # Should pass with 0.8 overall score
    
    @pytest.mark.asyncio
    async def test_evaluate_batch(self, benchmark_instance, sample_agent_config):
        """Test batch evaluation of scenarios."""
        # Create test scenarios
        scenarios = create_test_scenarios(3)
        
        # Mock agent
        mock_agent = AsyncMock(spec=BaseSkepticAgent)
        mock_agent.config = sample_agent_config
        
        # Mock evaluate_scenario to return successful results
        async def mock_evaluate_scenario(agent, scenario, context=None):
            from agent_skeptic_bench.models import EvaluationResult, SkepticResponse, EvaluationMetrics
            
            response = SkepticResponse(
                agent_id="test_agent",
                scenario_id=scenario.id,
                response_text="Test response",
                confidence_level=0.8,
                response_time_ms=1000
            )
            
            metrics = EvaluationMetrics(
                skepticism_calibration=0.8,
                evidence_standard_score=0.7,
                red_flag_detection=0.9,
                reasoning_quality=0.8,
                overall_score=0.8
            )
            
            return EvaluationResult(
                task_id=f"test_{scenario.id}",
                scenario=scenario,
                response=response,
                metrics=metrics,
                passed_evaluation=True
            )
        
        benchmark_instance.evaluate_scenario = mock_evaluate_scenario
        
        # Test batch evaluation
        results = await benchmark_instance.evaluate_batch(mock_agent, scenarios, concurrency=2)
        
        assert len(results) == 3
        assert all(result.passed_evaluation for result in results)
        assert all(result.scenario.id in [s.id for s in scenarios] for result in results)
    
    def test_create_session(self, benchmark_instance, sample_agent_config):
        """Test creating a benchmark session."""
        session = benchmark_instance.create_session(
            name="Test Session",
            agent_config=sample_agent_config,
            description="Test description"
        )
        
        assert session is not None
        assert session.name == "Test Session"
        assert session.description == "Test description"
        assert session.agent_config == sample_agent_config
        assert session.status == "running"
        assert session.id in benchmark_instance._active_sessions
    
    @pytest.mark.asyncio
    async def test_run_session(self, benchmark_instance, sample_agent_config):
        """Test running a complete benchmark session."""
        # Create session
        session = benchmark_instance.create_session(
            name="Test Session",
            agent_config=sample_agent_config
        )
        
        # Mock scenarios
        scenarios = create_test_scenarios(2)
        benchmark_instance.get_scenarios = Mock(return_value=scenarios)
        
        # Mock agent factory
        mock_agent = AsyncMock(spec=BaseSkepticAgent)
        benchmark_instance.agent_factory.create_agent = Mock(return_value=mock_agent)
        
        # Mock batch evaluation
        async def mock_evaluate_batch(agent, scenarios, concurrency):
            from agent_skeptic_bench.models import EvaluationResult, SkepticResponse, EvaluationMetrics
            
            results = []
            for scenario in scenarios:
                response = SkepticResponse(
                    agent_id="test_agent",
                    scenario_id=scenario.id,
                    response_text="Test response",
                    confidence_level=0.8,
                    response_time_ms=1000
                )
                
                metrics = EvaluationMetrics(
                    skepticism_calibration=0.8,
                    evidence_standard_score=0.7,
                    red_flag_detection=0.9,
                    reasoning_quality=0.8,
                    overall_score=0.8
                )
                
                result = EvaluationResult(
                    task_id=f"test_{scenario.id}",
                    scenario=scenario,
                    response=response,
                    metrics=metrics,
                    passed_evaluation=True
                )
                results.append(result)
            
            return results
        
        benchmark_instance.evaluate_batch = mock_evaluate_batch
        
        # Run session
        completed_session = await benchmark_instance.run_session(session, limit=2)
        
        assert completed_session.status == "completed"
        assert completed_session.total_scenarios == 2
        assert completed_session.passed_scenarios == 2
        assert completed_session.pass_rate == 1.0
        assert completed_session.completed_at is not None
    
    def test_compare_sessions(self, benchmark_instance):
        """Test comparing multiple sessions."""
        # Create mock sessions with metrics
        from agent_skeptic_bench.models import BenchmarkSession, EvaluationMetrics, AgentConfig, AgentProvider
        
        agent_config = AgentConfig(
            provider=AgentProvider.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        metrics1 = EvaluationMetrics(
            skepticism_calibration=0.8,
            evidence_standard_score=0.7,
            red_flag_detection=0.9,
            reasoning_quality=0.8,
            overall_score=0.8
        )
        
        metrics2 = EvaluationMetrics(
            skepticism_calibration=0.7,
            evidence_standard_score=0.8,
            red_flag_detection=0.8,
            reasoning_quality=0.7,
            overall_score=0.75
        )
        
        session1 = BenchmarkSession(
            id="session1",
            name="Session 1",
            agent_config=agent_config,
            summary_metrics=metrics1,
            status="completed"
        )
        session1._total_scenarios = 10
        session1._passed_scenarios = 8
        
        session2 = BenchmarkSession(
            id="session2", 
            name="Session 2",
            agent_config=agent_config,
            summary_metrics=metrics2,
            status="completed"
        )
        session2._total_scenarios = 10
        session2._passed_scenarios = 7
        
        benchmark_instance._active_sessions = {
            "session1": session1,
            "session2": session2
        }
        
        # Test comparison
        comparison = benchmark_instance.compare_sessions(["session1", "session2"])
        
        assert comparison["sessions"]
        assert len(comparison["sessions"]) == 2
        assert comparison["summary"]["total_sessions"] == 2
    
    def test_add_custom_scenario(self, benchmark_instance, sample_scenario):
        """Test adding custom scenarios."""
        benchmark_instance.scenario_loader.add_scenario = Mock()
        
        benchmark_instance.add_custom_scenario(sample_scenario)
        
        benchmark_instance.scenario_loader.add_scenario.assert_called_once_with(sample_scenario)
    
    def test_cleanup_session(self, benchmark_instance, sample_agent_config):
        """Test cleaning up sessions."""
        # Create session
        session = benchmark_instance.create_session(
            name="Test Session",
            agent_config=sample_agent_config
        )
        session_id = session.id
        
        # Verify session exists
        assert session_id in benchmark_instance._active_sessions
        
        # Clean up session
        result = benchmark_instance.cleanup_session(session_id)
        
        assert result == True
        assert session_id not in benchmark_instance._active_sessions
        
        # Test cleaning up non-existent session
        result = benchmark_instance.cleanup_session("non_existent")
        assert result == False