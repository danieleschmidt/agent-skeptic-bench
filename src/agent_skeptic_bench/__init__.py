"""Agent Skeptic Bench — benchmark for AI epistemic humility.

Tests whether AI agents know what they don't know:
calibration, uncertainty expression, and resistance to
overconfidence under adversarial pressure.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .scenarios import EpistemicScenario, ScenarioType, BENCHMARK_SCENARIOS
from .evaluator import SkepticismEvaluator, ModelResponse, CalibrationScore

__all__ = [
    "EpistemicScenario",
    "ScenarioType",
    "BENCHMARK_SCENARIOS",
    "SkepticismEvaluator",
    "ModelResponse",
    "CalibrationScore",
]
