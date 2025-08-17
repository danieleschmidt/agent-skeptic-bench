"""Agent Skeptic Bench - AI Agent Epistemic Vigilance Evaluation Framework.

A comprehensive framework for evaluating AI agents' ability to maintain appropriate 
skepticism and epistemic vigilance when encountering potentially misleading, 
fraudulent, or manipulative information.

Built with the Terragon Autonomous SDLC Value Enhancement System.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__email__ = "info@terragonlabs.com"

# Core components
from .agents import (
    AgentFactory,
    AnthropicSkepticAgent,
    BaseSkepticAgent,
    GoogleSkepticAgent,
    MockSkepticAgent,
    OpenAISkepticAgent,
    create_skeptic_agent,
)
from .benchmark import SkepticBenchmark
from .evaluation import EvaluationReport, run_full_evaluation

# Advanced features
from .features import AnalyticsDashboard, DataExporter, ReportGenerator, SearchEngine
from .models import (
    AgentConfig,
    AgentProvider,
    BenchmarkSession,
    EvaluationMetrics,
    EvaluationResult,
    Scenario,
    ScenarioCategory,
    SkepticResponse,
)
from .monitoring import HealthChecker, MetricsCollector, PerformanceMonitor

# Security and monitoring
from .security import AuditLogger, AuthenticationManager, InputValidator, RateLimiter

__all__ = [
    # Core
    "SkepticBenchmark",
    "AgentFactory",
    "BaseSkepticAgent",
    "OpenAISkepticAgent",
    "AnthropicSkepticAgent",
    "GoogleSkepticAgent",
    "MockSkepticAgent",
    "create_skeptic_agent",
    "run_full_evaluation",
    "EvaluationReport",

    # Models
    "Scenario",
    "ScenarioCategory",
    "SkepticResponse",
    "EvaluationMetrics",
    "EvaluationResult",
    "BenchmarkSession",
    "AgentConfig",
    "AgentProvider",

    # Advanced Features
    "SearchEngine",
    "ReportGenerator",
    "AnalyticsDashboard",
    "DataExporter",

    # Security
    "AuthenticationManager",
    "RateLimiter",
    "InputValidator",
    "AuditLogger",

    # Monitoring
    "MetricsCollector",
    "PerformanceMonitor",
    "HealthChecker",

    # Metadata
    "__version__",
]
