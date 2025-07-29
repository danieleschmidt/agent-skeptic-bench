"""Agent Skeptic Bench: Evaluating Epistemic Vigilance in AI Systems.

A comprehensive benchmark for testing AI agents' ability to maintain appropriate
skepticism and epistemic humility when evaluating claims and evidence.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"
__email__ = "skeptic-bench@yourdomain.com"

from .benchmark import SkepticBenchmark
from .agents import create_skeptic_agent
from .evaluation import run_full_evaluation

__all__ = [
    "SkepticBenchmark",
    "create_skeptic_agent", 
    "run_full_evaluation",
    "__version__",
]