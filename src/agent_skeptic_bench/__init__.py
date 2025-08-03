"""Agent Skeptic Bench: Evaluating Epistemic Vigilance in AI Systems.

A comprehensive benchmark for testing AI agents' ability to maintain appropriate
skepticism and epistemic humility when evaluating claims and evidence.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "skeptic-bench@yourdomain.com"

from .benchmark import SkepticBenchmark
from .agents import create_skeptic_agent, AgentFactory
from .evaluation import run_full_evaluation, compare_agents, EvaluationReport
from .scenarios import ScenarioLoader
from .metrics import MetricsCalculator
from .models import (
    Scenario, ScenarioCategory, AgentConfig, AgentProvider,
    SkepticResponse, EvaluationMetrics, EvaluationResult, BenchmarkSession
)

__all__ = [
    "SkepticBenchmark",
    "create_skeptic_agent",
    "AgentFactory", 
    "run_full_evaluation",
    "compare_agents",
    "EvaluationReport",
    "ScenarioLoader",
    "MetricsCalculator",
    "Scenario",
    "ScenarioCategory",
    "AgentConfig",
    "AgentProvider",
    "SkepticResponse",
    "EvaluationMetrics",
    "EvaluationResult",
    "BenchmarkSession",
    "__version__",
]