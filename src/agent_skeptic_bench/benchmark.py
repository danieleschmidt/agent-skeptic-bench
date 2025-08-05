"""Main benchmark class for Agent Skeptic Bench evaluation framework."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from .agents import BaseSkepticAgent, AgentFactory
from .scenarios import ScenarioLoader
from .metrics import MetricsCalculator
from .models import (
    AgentConfig,
    BenchmarkSession,
    EvaluationResult,
    EvaluationTask,
    Scenario,
    ScenarioCategory,
    SkepticResponse
)
from .exceptions import (
    EvaluationError,
    AgentTimeoutError,
    AgentResponseError,
    MetricsCalculationError,
    ScenarioNotFoundError
)
from .validation import response_validator
from .algorithms.optimization import QuantumInspiredOptimizer, SkepticismCalibrator


logger = logging.getLogger(__name__)


class SkepticBenchmark:
    """Main benchmark class for evaluating skeptical AI agents."""
    
    def __init__(self, 
                 scenario_loader: Optional[ScenarioLoader] = None,
                 agent_factory: Optional[AgentFactory] = None,
                 metrics_calculator: Optional[MetricsCalculator] = None):
        """Initialize the benchmark with optional custom components."""
        self.scenario_loader = scenario_loader or ScenarioLoader()
        self.agent_factory = agent_factory or AgentFactory()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self._active_sessions: Dict[str, BenchmarkSession] = {}
        self.quantum_calibrator = SkepticismCalibrator()
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get a specific scenario by ID."""
        return self.scenario_loader.get_scenario(scenario_id)
    
    def get_scenarios(self, categories: Optional[List[ScenarioCategory]] = None,
                     limit: Optional[int] = None,
                     difficulty: Optional[str] = None) -> List[Scenario]:
        """Get scenarios with optional filtering."""
        scenarios = self.scenario_loader.load_scenarios(categories)
        
        if difficulty:
            scenarios = [s for s in scenarios if s.metadata.get('difficulty') == difficulty]
        
        if limit:
            scenarios = scenarios[:limit]
        
        return scenarios
    
    async def evaluate_scenario(self,
                               skeptic_agent: BaseSkepticAgent,
                               scenario: Scenario,
                               context: Optional[Dict] = None,
                               timeout: Optional[float] = None) -> EvaluationResult:
        """Evaluate a single scenario with a skeptic agent."""
        start_time = datetime.utcnow()
        task_id = f"single_{scenario.id}_{start_time.timestamp()}"
        
        try:
            # Get skeptic response with timeout
            if timeout:
                response = await asyncio.wait_for(
                    skeptic_agent.evaluate_claim(scenario, context),
                    timeout=timeout
                )
            else:
                response = await skeptic_agent.evaluate_claim(scenario, context)
            
            # Validate response
            validation_errors = response_validator.validate_response(response)
            if validation_errors:
                logger.warning(f"Response validation issues for {scenario.id}: {validation_errors}")
                # Continue with evaluation but log issues
                
            # Calculate metrics with error handling
            try:
                metrics = self.metrics_calculator.calculate_metrics(response, scenario)
            except Exception as e:
                logger.error(f"Metrics calculation failed for {scenario.id}: {e}")
                raise MetricsCalculationError("overall_evaluation", str(e))
            
            # Create evaluation result
            result = EvaluationResult(
                task_id=task_id,
                scenario=scenario,
                response=response,
                metrics=metrics,
                analysis={
                    "single_evaluation": True,
                    "context": context or {},
                    "evaluation_duration_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    "validation_warnings": validation_errors
                },
                evaluation_notes=validation_errors
            )
            
            logger.info(f"Evaluated scenario {scenario.id}: Overall score {metrics.overall_score:.3f}")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Agent evaluation timed out after {timeout}s"
            logger.error(f"Timeout evaluating scenario {scenario.id}: {error_msg}")
            raise AgentTimeoutError(getattr(skeptic_agent, 'agent_id', 'unknown'), timeout or 0)
            
        except (AgentTimeoutError, AgentResponseError, MetricsCalculationError):
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error evaluating scenario {scenario.id}: {e}")
            raise EvaluationError(f"Failed to evaluate scenario {scenario.id}: {e}")
    
    async def evaluate_batch(self,
                            skeptic_agent: BaseSkepticAgent,
                            scenarios: List[Scenario],
                            concurrency: int = 5,
                            context: Optional[Dict] = None) -> List[EvaluationResult]:
        """Evaluate multiple scenarios concurrently."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def evaluate_with_semaphore(scenario: Scenario) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_scenario(skeptic_agent, scenario, context)
        
        logger.info(f"Starting batch evaluation of {len(scenarios)} scenarios with concurrency {concurrency}")
        results = await asyncio.gather(
            *[evaluate_with_semaphore(scenario) for scenario in scenarios],
            return_exceptions=True
        )
        
        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scenario {scenarios[i].id} failed: {result}")
                final_results.append(self._create_failed_result(scenarios[i], str(result)))
            else:
                final_results.append(result)
        
        logger.info(f"Completed batch evaluation: {len(final_results)} results")
        return final_results
    
    def create_session(self,
                      name: str,
                      agent_config: AgentConfig,
                      description: Optional[str] = None) -> BenchmarkSession:
        """Create a new benchmark session."""
        session = BenchmarkSession(
            name=name,
            description=description,
            agent_config=agent_config
        )
        
        self._active_sessions[session.id] = session
        logger.info(f"Created benchmark session: {session.id}")
        return session
    
    async def run_session(self,
                         session: BenchmarkSession,
                         categories: Optional[List[ScenarioCategory]] = None,
                         limit: Optional[int] = None,
                         concurrency: int = 5) -> BenchmarkSession:
        """Run a complete benchmark session."""
        try:
            session.status = "running"
            
            # Get scenarios
            scenarios = self.get_scenarios(categories or session.scenario_categories, limit)
            if not scenarios:
                raise ValueError("No scenarios available for evaluation")
            
            logger.info(f"Running session {session.id} with {len(scenarios)} scenarios")
            
            # Create agent
            agent = self.agent_factory.create_agent(session.agent_config)
            
            # Run evaluation
            results = await self.evaluate_batch(agent, scenarios, concurrency)
            
            # Add results to session
            for result in results:
                session.add_result(result)
            
            # Complete session
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            
            logger.info(f"Session {session.id} completed: {session.pass_rate:.1%} pass rate")
            return session
            
        except Exception as e:
            session.status = "failed"
            logger.error(f"Session {session.id} failed: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[BenchmarkSession]:
        """Get an active session by ID."""
        return self._active_sessions.get(session_id)
    
    def list_sessions(self) -> List[BenchmarkSession]:
        """List all active sessions."""
        return list(self._active_sessions.values())
    
    def compare_sessions(self, session_ids: List[str]) -> Dict:
        """Compare metrics across multiple sessions."""
        sessions = [self._active_sessions.get(sid) for sid in session_ids]
        sessions = [s for s in sessions if s is not None]
        
        if not sessions:
            return {"error": "No valid sessions found"}
        
        comparison = {
            "sessions": [],
            "summary": {
                "total_sessions": len(sessions),
                "categories_covered": set(),
                "total_scenarios": 0
            }
        }
        
        for session in sessions:
            if not session.summary_metrics:
                continue
                
            session_data = {
                "id": session.id,
                "name": session.name,
                "agent_model": session.agent_config.model_name,
                "total_scenarios": session.total_scenarios,
                "pass_rate": session.pass_rate,
                "metrics": {
                    "skepticism_calibration": session.summary_metrics.skepticism_calibration,
                    "evidence_standard_score": session.summary_metrics.evidence_standard_score,
                    "red_flag_detection": session.summary_metrics.red_flag_detection,
                    "reasoning_quality": session.summary_metrics.reasoning_quality,
                    "overall_score": session.summary_metrics.overall_score
                }
            }
            
            comparison["sessions"].append(session_data)
            comparison["summary"]["total_scenarios"] += session.total_scenarios
            
            # Collect categories
            for result in session.results:
                comparison["summary"]["categories_covered"].add(result.scenario.category.value)
        
        comparison["summary"]["categories_covered"] = list(comparison["summary"]["categories_covered"])
        return comparison
    
    def generate_leaderboard(self, 
                           category: Optional[ScenarioCategory] = None,
                           limit: int = 10) -> Dict:
        """Generate leaderboard from all sessions."""
        all_sessions = self.list_sessions()
        completed_sessions = [s for s in all_sessions if s.status == "completed" and s.summary_metrics]
        
        if category:
            # Filter sessions that have results for the specific category
            filtered_sessions = []
            for session in completed_sessions:
                category_results = [r for r in session.results if r.scenario.category == category]
                if category_results:
                    filtered_sessions.append(session)
            completed_sessions = filtered_sessions
        
        # Sort by overall score
        leaderboard_entries = []
        for session in completed_sessions:
            if category:
                # Calculate category-specific metrics
                category_results = [r for r in session.results if r.scenario.category == category]
                if category_results:
                    avg_metrics = self._calculate_average_metrics(category_results)
                    overall_score = avg_metrics.overall_score
                else:
                    continue
            else:
                overall_score = session.summary_metrics.overall_score
            
            leaderboard_entries.append({
                "rank": 0,  # Will be set after sorting
                "session_id": session.id,
                "session_name": session.name,
                "agent_model": session.agent_config.model_name,
                "overall_score": overall_score,
                "pass_rate": session.pass_rate,
                "total_scenarios": session.total_scenarios,
                "completed_at": session.completed_at.isoformat() if session.completed_at else None
            })
        
        # Sort by overall score (descending)
        leaderboard_entries.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard_entries[:limit]):
            entry["rank"] = i + 1
        
        return {
            "category": category.value if category else "all",
            "total_entries": len(leaderboard_entries),
            "leaderboard": leaderboard_entries[:limit],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _create_failed_result(self, scenario: Scenario, error_message: str) -> EvaluationResult:
        """Create a failed evaluation result."""
        from .models import EvaluationMetrics
        
        failed_response = SkepticResponse(
            agent_id="failed",
            scenario_id=scenario.id,
            response_text=f"Evaluation failed: {error_message}",
            confidence_level=0.0,
            response_time_ms=0
        )
        
        failed_metrics = EvaluationMetrics(
            skepticism_calibration=0.0,
            evidence_standard_score=0.0,
            red_flag_detection=0.0,
            reasoning_quality=0.0,
            overall_score=0.0
        )
        
        return EvaluationResult(
            task_id=f"failed_{scenario.id}_{datetime.utcnow().timestamp()}",
            scenario=scenario,
            response=failed_response,
            metrics=failed_metrics,
            analysis={"error": error_message},
            passed_evaluation=False,
            evaluation_notes=[f"Failed: {error_message}"]
        )
    
    def _calculate_average_metrics(self, results: List[EvaluationResult]):
        """Calculate average metrics from a list of results."""
        from .models import EvaluationMetrics
        
        if not results:
            return EvaluationMetrics(
                skepticism_calibration=0.0,
                evidence_standard_score=0.0,
                red_flag_detection=0.0,
                reasoning_quality=0.0,
                overall_score=0.0
            )
        
        metrics = [r.metrics for r in results]
        return EvaluationMetrics(
            skepticism_calibration=sum(m.skepticism_calibration for m in metrics) / len(metrics),
            evidence_standard_score=sum(m.evidence_standard_score for m in metrics) / len(metrics),
            red_flag_detection=sum(m.red_flag_detection for m in metrics) / len(metrics),
            reasoning_quality=sum(m.reasoning_quality for m in metrics) / len(metrics)
        )
    
    def add_custom_scenario(self, scenario: Scenario) -> None:
        """Add a custom scenario to the benchmark."""
        self.scenario_loader.add_scenario(scenario)
        logger.info(f"Added custom scenario: {scenario.id}")
    
    def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from memory."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.info(f"Cleaned up session: {session_id}")
            return True
        return False
    
    def optimize_agent_parameters(self, 
                                 session_id: str, 
                                 target_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Optimize agent parameters using quantum-inspired algorithms."""
        session = self._active_sessions.get(session_id)
        if not session or not session.results:
            raise ValueError(f"No valid session found with results: {session_id}")
        
        # Convert session results to optimization format
        evaluation_data = []
        for result in session.results:
            evaluation_data.append((result.scenario, result.response, result.metrics))
        
        # Run quantum optimization
        optimal_params = self.quantum_calibrator.calibrate_agent_parameters(
            evaluation_data, target_metrics
        )
        
        logger.info(f"Optimized parameters for session {session_id}: {optimal_params}")
        return optimal_params
    
    def predict_scenario_difficulty(self, scenario: Scenario, agent_params: Dict[str, float]) -> float:
        """Predict scenario difficulty using quantum uncertainty principles."""
        predicted_skepticism = self.quantum_calibrator.predict_optimal_skepticism(
            scenario, agent_params
        )
        
        # Difficulty correlates with uncertainty in optimal skepticism
        base_difficulty = abs(predicted_skepticism - 0.5) * 2  # Distance from neutral
        complexity_adjustment = len(scenario.description) / 1000.0  # Text complexity
        
        difficulty = (base_difficulty + complexity_adjustment) / 2
        return max(0.0, min(1.0, difficulty))
    
    def get_quantum_insights(self, session_id: str) -> Dict[str, any]:
        """Get quantum-inspired insights for evaluation session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if not session.results:
            return {"error": "No results available"}
        
        # Analyze quantum coherence across evaluations
        coherence_scores = []
        entanglement_measures = []
        
        for result in session.results:
            # Calculate quantum coherence
            expected = result.scenario.correct_skepticism_level
            actual = result.response.confidence_level
            coherence = 1.0 - abs(expected - actual)
            coherence_scores.append(coherence)
            
            # Measure parameter entanglement
            param_correlation = self._calculate_parameter_entanglement(result)
            entanglement_measures.append(param_correlation)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_entanglement = sum(entanglement_measures) / len(entanglement_measures)
        
        return {
            "quantum_coherence": avg_coherence,
            "parameter_entanglement": avg_entanglement,
            "coherence_distribution": coherence_scores,
            "entanglement_distribution": entanglement_measures,
            "optimization_recommendations": self._generate_quantum_recommendations(
                avg_coherence, avg_entanglement
            )
        }
    
    def _calculate_parameter_entanglement(self, result: EvaluationResult) -> float:
        """Calculate quantum entanglement measure for evaluation parameters."""
        # Use metrics as proxy for parameter entanglement
        metrics_values = [
            result.metrics.skepticism_calibration,
            result.metrics.evidence_standard_score,
            result.metrics.red_flag_detection,
            result.metrics.reasoning_quality
        ]
        
        # Calculate correlations between metrics
        correlations = []
        for i in range(len(metrics_values)):
            for j in range(i + 1, len(metrics_values)):
                correlation = abs(metrics_values[i] * metrics_values[j])
                correlations.append(correlation)
        
        return sum(correlations) / len(correlations) if correlations else 0.0
    
    def _generate_quantum_recommendations(self, coherence: float, entanglement: float) -> List[str]:
        """Generate recommendations based on quantum analysis."""
        recommendations = []
        
        if coherence < 0.3:
            recommendations.append("Low quantum coherence detected - agent responses may be inconsistent")
        elif coherence > 0.8:
            recommendations.append("High quantum coherence - agent shows good skepticism alignment")
        
        if entanglement < 0.2:
            recommendations.append("Low parameter entanglement - metrics may be operating independently")
        elif entanglement > 0.7:
            recommendations.append("High parameter entanglement - strong correlation between evaluation metrics")
        
        if coherence > 0.6 and entanglement > 0.5:
            recommendations.append("Optimal quantum state achieved - agent is well-calibrated")
        
        return recommendations if recommendations else ["Quantum analysis complete - no specific recommendations"]