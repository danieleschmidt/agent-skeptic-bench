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
from .benchmark import SkepticalEvaluationBenchmark
from .agents import AgentFactory, OpenAIAgent, AnthropicAgent, GoogleAIAgent
from .models import (
    Scenario, ScenarioCategory, 
    SkepticResponse, EvaluationMetrics, EvaluationResult, 
    BenchmarkSession
)

# Advanced features
from .features import (
    SearchEngine, ReportGenerator, AnalyticsDashboard, DataExporter
)

# Security and monitoring
from .security import (
    AuthenticationManager, RateLimiter, InputValidator, AuditLogger
)

from .monitoring import (
    MetricsCollector, PerformanceMonitor, HealthChecker
)

__all__ = [
    # Core
    "SkepticalEvaluationBenchmark",
    "AgentFactory",
    "OpenAIAgent",
    "AnthropicAgent",
    "GoogleAIAgent",
    
    # Models
    "Scenario",
    "ScenarioCategory",
    "SkepticResponse",
    "EvaluationMetrics", 
    "EvaluationResult",
    "BenchmarkSession",
    
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