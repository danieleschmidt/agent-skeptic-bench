"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..models import AgentProvider, ScenarioCategory


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema."""
    class Config:
        from_attributes = True


# Scenario schemas
class ScenarioBase(BaseModel):
    """Base scenario schema."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    adversary_claim: str = Field(..., min_length=1)
    correct_skepticism_level: float = Field(..., ge=0.0, le=1.0)
    good_evidence_requests: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioCreate(ScenarioBase):
    """Schema for creating scenarios."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    category: ScenarioCategory


class ScenarioUpdate(BaseModel):
    """Schema for updating scenarios."""
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, min_length=1)
    adversary_claim: str | None = Field(None, min_length=1)
    category: ScenarioCategory | None = None
    correct_skepticism_level: float | None = Field(None, ge=0.0, le=1.0)
    good_evidence_requests: list[str] | None = None
    red_flags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ScenarioResponse(ScenarioBase, BaseResponse):
    """Schema for scenario responses."""
    id: str
    category: ScenarioCategory
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record):
        """Create from database record."""
        return cls(
            id=record.id,
            category=ScenarioCategory(record.category),
            name=record.name,
            description=record.description,
            adversary_claim=record.adversary_claim,
            correct_skepticism_level=record.correct_skepticism_level,
            good_evidence_requests=record.good_evidence_requests or [],
            red_flags=record.red_flags or [],
            metadata=record.metadata or {},
            created_at=record.created_at,
            updated_at=record.updated_at
        )

    def to_domain_model(self):
        """Convert to domain model."""
        from ..models import Scenario
        return Scenario(
            id=self.id,
            category=self.category,
            name=self.name,
            description=self.description,
            adversary_claim=self.adversary_claim,
            correct_skepticism_level=self.correct_skepticism_level,
            good_evidence_requests=self.good_evidence_requests,
            red_flags=self.red_flags,
            metadata=self.metadata,
            created_at=self.created_at
        )


# Agent schemas
class AgentConfigRequest(BaseModel):
    """Schema for agent configuration requests."""
    provider: AgentProvider
    model_name: str = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    timeout: int = Field(default=30, gt=0)
    retry_attempts: int = Field(default=3, ge=0)


class AgentResponse(BaseModel):
    """Schema for agent responses."""
    agent_id: str
    provider: AgentProvider
    model_name: str
    available: bool
    last_checked: datetime


# Evaluation schemas
class EvaluationRequest(BaseModel):
    """Schema for evaluation requests."""
    scenario_id: str
    agent_config: AgentConfigRequest
    context: dict[str, Any] | None = None


class BatchEvaluationRequest(BaseModel):
    """Schema for batch evaluation requests."""
    scenario_ids: list[str] = Field(..., min_items=1)
    agent_config: AgentConfigRequest
    concurrency: int = Field(default=5, ge=1, le=10)
    context: dict[str, Any] | None = None


class SkepticResponseSchema(BaseModel):
    """Schema for skeptic responses."""
    agent_id: str
    scenario_id: str
    response_text: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    evidence_requests: list[str]
    red_flags_identified: list[str]
    reasoning_steps: list[str]
    response_time_ms: int
    timestamp: datetime


class EvaluationMetricsSchema(BaseModel):
    """Schema for evaluation metrics."""
    skepticism_calibration: float = Field(ge=0.0, le=1.0)
    evidence_standard_score: float = Field(ge=0.0, le=1.0)
    red_flag_detection: float = Field(ge=0.0, le=1.0)
    reasoning_quality: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvaluationResponse(BaseResponse):
    """Schema for evaluation responses."""
    id: str
    task_id: str
    scenario: ScenarioResponse
    response: SkepticResponseSchema
    metrics: EvaluationMetricsSchema
    analysis: dict[str, Any]
    passed_evaluation: bool
    evaluation_notes: list[str]
    evaluated_at: datetime

    @classmethod
    def from_record(cls, record):
        """Create from database record."""
        return cls(
            id=record.id,
            task_id=record.task_id,
            scenario=ScenarioResponse.from_record(record.scenario),
            response=SkepticResponseSchema(
                agent_id=record.agent_id,
                scenario_id=record.scenario_id,
                response_text=record.response_text,
                confidence_level=record.confidence_level,
                evidence_requests=record.evidence_requests or [],
                red_flags_identified=record.red_flags_identified or [],
                reasoning_steps=record.reasoning_steps or [],
                response_time_ms=record.response_time_ms,
                timestamp=record.evaluated_at
            ),
            metrics=EvaluationMetricsSchema(
                skepticism_calibration=record.skepticism_calibration,
                evidence_standard_score=record.evidence_standard_score,
                red_flag_detection=record.red_flag_detection,
                reasoning_quality=record.reasoning_quality,
                overall_score=record.overall_score
            ),
            analysis=record.analysis or {},
            passed_evaluation=record.passed_evaluation,
            evaluation_notes=record.evaluation_notes or [],
            evaluated_at=record.evaluated_at
        )


# Session schemas
class SessionCreate(BaseModel):
    """Schema for creating sessions."""
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    agent_config: AgentConfigRequest
    scenario_categories: list[ScenarioCategory] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    """Schema for updating sessions."""
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    status: str | None = None


class SessionResponse(BaseResponse):
    """Schema for session responses."""
    id: str
    name: str
    description: str | None
    agent_provider: AgentProvider
    agent_model: str
    scenario_categories: list[ScenarioCategory]
    status: str
    started_at: datetime
    completed_at: datetime | None
    total_scenarios: int
    passed_scenarios: int
    pass_rate: float
    overall_score: float | None

    @classmethod
    def from_record(cls, record):
        """Create from database record."""
        return cls(
            id=record.id,
            name=record.name,
            description=record.description,
            agent_provider=AgentProvider(record.agent_provider),
            agent_model=record.agent_model,
            scenario_categories=[ScenarioCategory(cat) for cat in (record.scenario_categories or [])],
            status=record.status,
            started_at=record.started_at,
            completed_at=record.completed_at,
            total_scenarios=record.total_scenarios,
            passed_scenarios=record.passed_scenarios,
            pass_rate=record.pass_rate,
            overall_score=record.overall_score
        )


class SessionDetailResponse(SessionResponse):
    """Detailed session response with metrics."""
    avg_skepticism_calibration: float | None
    avg_evidence_standard_score: float | None
    avg_red_flag_detection: float | None
    avg_reasoning_quality: float | None
    evaluation_count: int = 0

    @classmethod
    def from_record(cls, record):
        """Create from database record."""
        base = SessionResponse.from_record(record)
        return cls(
            **base.dict(),
            avg_skepticism_calibration=record.avg_skepticism_calibration,
            avg_evidence_standard_score=record.avg_evidence_standard_score,
            avg_red_flag_detection=record.avg_red_flag_detection,
            avg_reasoning_quality=record.avg_reasoning_quality,
            evaluation_count=len(record.evaluations) if hasattr(record, 'evaluations') and record.evaluations else 0
        )


# Comparison schemas
class ComparisonRequest(BaseModel):
    """Schema for comparison requests."""
    session_ids: list[str] = Field(..., min_items=2)
    category: ScenarioCategory | None = None


class LeaderboardRequest(BaseModel):
    """Schema for leaderboard requests."""
    category: ScenarioCategory | None = None
    limit: int = Field(default=10, ge=1, le=100)


class LeaderboardEntry(BaseModel):
    """Schema for leaderboard entries."""
    rank: int
    session_id: str
    session_name: str
    agent_model: str
    agent_provider: AgentProvider
    overall_score: float
    pass_rate: float
    total_scenarios: int
    completed_at: datetime | None


class LeaderboardResponse(BaseResponse):
    """Schema for leaderboard responses."""
    category: str | None
    total_entries: int
    leaderboard: list[LeaderboardEntry]
    generated_at: datetime


# Error schemas
class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    message: str
    type: str
    details: dict[str, Any] | None = None


class ValidationErrorResponse(ErrorResponse):
    """Schema for validation error responses."""
    type: str = "validation_error"
    details: dict[str, list[str]]


# Health check schemas
class HealthResponse(BaseModel):
    """Schema for health check responses."""
    status: str
    timestamp: datetime
    version: str
    database: dict[str, Any]
    cache: dict[str, Any]
