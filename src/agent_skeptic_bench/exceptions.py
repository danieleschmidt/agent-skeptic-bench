"""Custom exceptions for Agent Skeptic Bench."""

from typing import Any


class AgentSkepticBenchError(Exception):
    """Base exception for Agent Skeptic Bench."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ScenarioError(AgentSkepticBenchError):
    """Exception raised for scenario-related errors."""
    pass


class ScenarioNotFoundError(ScenarioError):
    """Exception raised when a scenario is not found."""

    def __init__(self, scenario_id: str):
        super().__init__(
            f"Scenario not found: {scenario_id}",
            {"scenario_id": scenario_id}
        )
        self.scenario_id = scenario_id


class InvalidScenarioError(ScenarioError):
    """Exception raised when a scenario is malformed or invalid."""

    def __init__(self, scenario_id: str, validation_errors: list):
        super().__init__(
            f"Invalid scenario: {scenario_id}",
            {"scenario_id": scenario_id, "validation_errors": validation_errors}
        )
        self.scenario_id = scenario_id
        self.validation_errors = validation_errors


class AgentError(AgentSkepticBenchError):
    """Exception raised for agent-related errors."""
    pass


class AgentTimeoutError(AgentError):
    """Exception raised when agent evaluation times out."""

    def __init__(self, agent_id: str, timeout_seconds: float):
        super().__init__(
            f"Agent {agent_id} timed out after {timeout_seconds} seconds",
            {"agent_id": agent_id, "timeout_seconds": timeout_seconds}
        )
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds


class AgentResponseError(AgentError):
    """Exception raised when agent response is invalid."""

    def __init__(self, agent_id: str, response_errors: list):
        super().__init__(
            f"Invalid response from agent {agent_id}",
            {"agent_id": agent_id, "response_errors": response_errors}
        )
        self.agent_id = agent_id
        self.response_errors = response_errors


class EvaluationError(AgentSkepticBenchError):
    """Exception raised for evaluation-related errors."""
    pass


class MetricsCalculationError(EvaluationError):
    """Exception raised when metrics calculation fails."""

    def __init__(self, metric_name: str, error_details: str):
        super().__init__(
            f"Failed to calculate metric '{metric_name}': {error_details}",
            {"metric_name": metric_name, "error_details": error_details}
        )
        self.metric_name = metric_name
        self.error_details = error_details


class DataError(AgentSkepticBenchError):
    """Exception raised for data-related errors."""
    pass


class DataLoadError(DataError):
    """Exception raised when data loading fails."""

    def __init__(self, file_path: str, error_details: str):
        super().__init__(
            f"Failed to load data from {file_path}: {error_details}",
            {"file_path": file_path, "error_details": error_details}
        )
        self.file_path = file_path
        self.error_details = error_details


class ConfigurationError(AgentSkepticBenchError):
    """Exception raised for configuration-related errors."""
    pass


class SecurityError(AgentSkepticBenchError):
    """Exception raised for security-related errors."""
    pass


class RateLimitError(SecurityError):
    """Exception raised when rate limits are exceeded."""

    def __init__(self, identifier: str, limit: int, window_seconds: int):
        super().__init__(
            f"Rate limit exceeded for {identifier}: {limit} requests per {window_seconds} seconds",
            {"identifier": identifier, "limit": limit, "window_seconds": window_seconds}
        )
        self.identifier = identifier
        self.limit = limit
        self.window_seconds = window_seconds


class ValidationError(AgentSkepticBenchError):
    """Exception raised for validation errors."""

    def __init__(self, field_name: str, invalid_value: Any, expected: str):
        super().__init__(
            f"Validation failed for field '{field_name}': got {invalid_value}, expected {expected}",
            {"field_name": field_name, "invalid_value": invalid_value, "expected": expected}
        )
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected = expected
