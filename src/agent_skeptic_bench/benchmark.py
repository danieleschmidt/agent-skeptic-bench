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