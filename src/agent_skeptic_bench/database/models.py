"""SQLAlchemy models for Agent Skeptic Bench database."""

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ..models import AgentProvider, ScenarioCategory

Base = declarative_base()


class ScenarioRecord(Base):
    """Database model for evaluation scenarios."""

    __tablename__ = "scenarios"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    adversary_claim: Mapped[str] = mapped_column(Text, nullable=False)
    correct_skepticism_level: Mapped[float] = mapped_column(Float, nullable=False)
    good_evidence_requests: Mapped[list[str]] = mapped_column(JSON, default=list)
    red_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    scenario_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    evaluations = relationship("EvaluationRecord", back_populates="scenario")

    def __repr__(self) -> str:
        return f"<ScenarioRecord(id='{self.id}', name='{self.name}', category='{self.category}')>"

    def to_domain_model(self):
        """Convert to domain model."""
        from ..models import Scenario

        return Scenario(
            id=self.id,
            category=ScenarioCategory(self.category),
            name=self.name,
            description=self.description,
            adversary_claim=self.adversary_claim,
            correct_skepticism_level=self.correct_skepticism_level,
            good_evidence_requests=self.good_evidence_requests or [],
            red_flags=self.red_flags or [],
            metadata=self.scenario_metadata or {},
            created_at=self.created_at
        )


class SessionRecord(Base):
    """Database model for benchmark sessions."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Agent configuration
    agent_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    agent_model: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # Session metadata
    scenario_categories: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="running", index=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Summary metrics (calculated after completion)
    total_scenarios: Mapped[int] = mapped_column(Integer, default=0)
    passed_scenarios: Mapped[int] = mapped_column(Integer, default=0)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Overall metrics
    avg_skepticism_calibration: Mapped[float | None] = mapped_column(Float)
    avg_evidence_standard_score: Mapped[float | None] = mapped_column(Float)
    avg_red_flag_detection: Mapped[float | None] = mapped_column(Float)
    avg_reasoning_quality: Mapped[float | None] = mapped_column(Float)
    overall_score: Mapped[float | None] = mapped_column(Float)

    # Relationships
    evaluations = relationship("EvaluationRecord", back_populates="session")

    def __repr__(self) -> str:
        return f"<SessionRecord(id='{self.id}', name='{self.name}', status='{self.status}')>"

    def to_domain_model(self):
        """Convert to domain model."""
        from ..models import AgentConfig, BenchmarkSession, EvaluationMetrics

        agent_config = AgentConfig(
            provider=AgentProvider(self.agent_provider),
            model_name=self.agent_model,
            api_key="***",  # Don't expose in domain model
            **self.agent_config
        )

        summary_metrics = None
        if self.overall_score is not None:
            summary_metrics = EvaluationMetrics(
                skepticism_calibration=self.avg_skepticism_calibration or 0.0,
                evidence_standard_score=self.avg_evidence_standard_score or 0.0,
                red_flag_detection=self.avg_red_flag_detection or 0.0,
                reasoning_quality=self.avg_reasoning_quality or 0.0,
                overall_score=self.overall_score
            )

        scenario_categories = [ScenarioCategory(cat) for cat in (self.scenario_categories or [])]

        return BenchmarkSession(
            id=self.id,
            name=self.name,
            description=self.description,
            agent_config=agent_config,
            scenario_categories=scenario_categories,
            summary_metrics=summary_metrics,
            started_at=self.started_at,
            completed_at=self.completed_at,
            status=self.status
        )


class EvaluationRecord(Base):
    """Database model for evaluation results."""

    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(50), sa.ForeignKey("sessions.id"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(50), sa.ForeignKey("scenarios.id"), index=True)

    # Agent response
    agent_id: Mapped[str] = mapped_column(String(100), nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_level: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_requests: Mapped[list[str]] = mapped_column(JSON, default=list)
    red_flags_identified: Mapped[list[str]] = mapped_column(JSON, default=list)
    reasoning_steps: Mapped[list[str]] = mapped_column(JSON, default=list)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Evaluation metrics
    skepticism_calibration: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_standard_score: Mapped[float] = mapped_column(Float, nullable=False)
    red_flag_detection: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning_quality: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Result metadata
    passed_evaluation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evaluation_notes: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Timestamps
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("SessionRecord", back_populates="evaluations")
    scenario = relationship("ScenarioRecord", back_populates="evaluations")

    # Indexes for performance
    __table_args__ = (
        sa.Index("idx_evaluations_session_scenario", "session_id", "scenario_id"),
        sa.Index("idx_evaluations_overall_score", "overall_score"),
        sa.Index("idx_evaluations_passed", "passed_evaluation"),
        sa.Index("idx_evaluations_evaluated_at", "evaluated_at"),
    )

    def __repr__(self) -> str:
        return f"<EvaluationRecord(id='{self.id}', scenario_id='{self.scenario_id}', score={self.overall_score:.3f})>"

    def to_domain_model(self):
        """Convert to domain model."""
        from ..models import EvaluationMetrics, EvaluationResult, SkepticResponse

        # Create response object
        response = SkepticResponse(
            agent_id=self.agent_id,
            scenario_id=self.scenario_id,
            response_text=self.response_text,
            confidence_level=self.confidence_level,
            evidence_requests=self.evidence_requests or [],
            red_flags_identified=self.red_flags_identified or [],
            reasoning_steps=self.reasoning_steps or [],
            response_time_ms=self.response_time_ms,
            timestamp=self.evaluated_at
        )

        # Create metrics object
        metrics = EvaluationMetrics(
            skepticism_calibration=self.skepticism_calibration,
            evidence_standard_score=self.evidence_standard_score,
            red_flag_detection=self.red_flag_detection,
            reasoning_quality=self.reasoning_quality,
            overall_score=self.overall_score
        )

        # Get scenario from relationship or create minimal one
        scenario = self.scenario.to_domain_model() if self.scenario else None

        return EvaluationResult(
            id=self.id,
            task_id=self.task_id,
            scenario=scenario,
            response=response,
            metrics=metrics,
            analysis=self.analysis or {},
            passed_evaluation=self.passed_evaluation,
            evaluation_notes=self.evaluation_notes or [],
            evaluated_at=self.evaluated_at
        )


class CacheEntry(Base):
    """Database model for caching evaluation results."""

    __tablename__ = "cache_entries"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Hash of cache key
    key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True, index=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<CacheEntry(key='{self.key[:50]}...', expires_at='{self.expires_at}')>"

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at.replace(tzinfo=None)


class AuditLog(Base):
    """Database model for audit logging."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(String(100), index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(100), index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', resource='{self.resource_type}')>"
