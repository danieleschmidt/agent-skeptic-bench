"""Repository classes for database operations."""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import (
    Scenario, BenchmarkSession, EvaluationResult, 
    ScenarioCategory, AgentProvider
)
from .models import (
    ScenarioRecord, SessionRecord, EvaluationRecord, 
    CacheEntry, AuditLog
)


logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common functionality."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        await self.session.rollback()


class ScenarioRepository(BaseRepository):
    """Repository for scenario operations."""
    
    async def create(self, scenario: Scenario) -> ScenarioRecord:
        """Create a new scenario record."""
        record = ScenarioRecord(
            id=scenario.id,
            category=scenario.category.value,
            name=scenario.name,
            description=scenario.description,
            adversary_claim=scenario.adversary_claim,
            correct_skepticism_level=scenario.correct_skepticism_level,
            good_evidence_requests=scenario.good_evidence_requests,
            red_flags=scenario.red_flags,
            metadata=scenario.metadata
        )
        
        self.session.add(record)
        await self.session.flush()
        logger.debug(f"Created scenario record: {scenario.id}")
        return record
    
    async def get_by_id(self, scenario_id: str) -> Optional[ScenarioRecord]:
        """Get scenario by ID."""
        result = await self.session.execute(
            sa.select(ScenarioRecord).where(ScenarioRecord.id == scenario_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_category(self, category: ScenarioCategory) -> List[ScenarioRecord]:
        """Get all scenarios in a category."""
        result = await self.session.execute(
            sa.select(ScenarioRecord).where(ScenarioRecord.category == category.value)
        )
        return list(result.scalars().all())
    
    async def get_all(self, 
                     categories: Optional[List[ScenarioCategory]] = None,
                     limit: Optional[int] = None,
                     offset: int = 0) -> List[ScenarioRecord]:
        """Get all scenarios with optional filtering."""
        query = sa.select(ScenarioRecord)
        
        if categories:
            category_values = [cat.value for cat in categories]
            query = query.where(ScenarioRecord.category.in_(category_values))
        
        query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def search(self, 
                    query_text: str,
                    categories: Optional[List[ScenarioCategory]] = None) -> List[ScenarioRecord]:
        """Search scenarios by text in name or description."""
        query = sa.select(ScenarioRecord).where(
            sa.or_(
                ScenarioRecord.name.ilike(f"%{query_text}%"),
                ScenarioRecord.description.ilike(f"%{query_text}%")
            )
        )
        
        if categories:
            category_values = [cat.value for cat in categories]
            query = query.where(ScenarioRecord.category.in_(category_values))
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update(self, scenario_id: str, updates: Dict[str, Any]) -> Optional[ScenarioRecord]:
        """Update scenario record."""
        record = await self.get_by_id(scenario_id)
        if not record:
            return None
        
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        await self.session.flush()
        logger.debug(f"Updated scenario record: {scenario_id}")
        return record
    
    async def delete(self, scenario_id: str) -> bool:
        """Delete scenario record."""
        record = await self.get_by_id(scenario_id)
        if not record:
            return False
        
        await self.session.delete(record)
        await self.session.flush()
        logger.debug(f"Deleted scenario record: {scenario_id}")
        return True
    
    async def count_by_category(self) -> Dict[str, int]:
        """Count scenarios by category."""
        result = await self.session.execute(
            sa.select(ScenarioRecord.category, sa.func.count(ScenarioRecord.id))
            .group_by(ScenarioRecord.category)
        )
        return dict(result.all())


class SessionRepository(BaseRepository):
    """Repository for session operations."""
    
    async def create(self, session: BenchmarkSession) -> SessionRecord:
        """Create a new session record."""
        # Extract agent configuration
        agent_config = {
            "temperature": session.agent_config.temperature,
            "max_tokens": session.agent_config.max_tokens,
            "timeout": session.agent_config.timeout,
            "retry_attempts": session.agent_config.retry_attempts
        }
        
        record = SessionRecord(
            id=session.id,
            name=session.name,
            description=session.description,
            agent_provider=session.agent_config.provider.value,
            agent_model=session.agent_config.model_name,
            agent_config=agent_config,
            scenario_categories=[cat.value for cat in session.scenario_categories],
            status=session.status,
            started_at=session.started_at
        )
        
        self.session.add(record)
        await self.session.flush()
        logger.debug(f"Created session record: {session.id}")
        return record
    
    async def get_by_id(self, session_id: str, include_evaluations: bool = False) -> Optional[SessionRecord]:
        """Get session by ID."""
        query = sa.select(SessionRecord).where(SessionRecord.id == session_id)
        
        if include_evaluations:
            query = query.options(selectinload(SessionRecord.evaluations))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(self, 
                     status: Optional[str] = None,
                     limit: Optional[int] = None,
                     offset: int = 0) -> List[SessionRecord]:
        """Get all sessions with optional filtering."""
        query = sa.select(SessionRecord)
        
        if status:
            query = query.where(SessionRecord.status == status)
        
        query = query.order_by(SessionRecord.created_at.desc())
        query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_status(self, session_id: str, status: str, completed_at: Optional[datetime] = None) -> Optional[SessionRecord]:
        """Update session status."""
        record = await self.get_by_id(session_id)
        if not record:
            return None
        
        record.status = status
        if completed_at:
            record.completed_at = completed_at
        
        await self.session.flush()
        logger.debug(f"Updated session status: {session_id} -> {status}")
        return record
    
    async def update_metrics(self, session_id: str, 
                           total_scenarios: int,
                           passed_scenarios: int,
                           metrics: Dict[str, float]) -> Optional[SessionRecord]:
        """Update session summary metrics."""
        record = await self.get_by_id(session_id)
        if not record:
            return None
        
        record.total_scenarios = total_scenarios
        record.passed_scenarios = passed_scenarios
        record.pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        
        record.avg_skepticism_calibration = metrics.get("skepticism_calibration")
        record.avg_evidence_standard_score = metrics.get("evidence_standard_score")
        record.avg_red_flag_detection = metrics.get("red_flag_detection")
        record.avg_reasoning_quality = metrics.get("reasoning_quality")
        record.overall_score = metrics.get("overall_score")
        
        await self.session.flush()
        logger.debug(f"Updated session metrics: {session_id}")
        return record
    
    async def delete(self, session_id: str) -> bool:
        """Delete session and all its evaluations."""
        record = await self.get_by_id(session_id)
        if not record:
            return False
        
        # Delete associated evaluations first
        await self.session.execute(
            sa.delete(EvaluationRecord).where(EvaluationRecord.session_id == session_id)
        )
        
        await self.session.delete(record)
        await self.session.flush()
        logger.debug(f"Deleted session record: {session_id}")
        return True
    
    async def get_leaderboard(self, 
                            category: Optional[ScenarioCategory] = None,
                            limit: int = 10) -> List[SessionRecord]:
        """Get leaderboard of sessions by overall score."""
        query = sa.select(SessionRecord).where(
            SessionRecord.status == "completed",
            SessionRecord.overall_score.is_not(None)
        )
        
        # If category specified, filter sessions that have evaluations in that category
        if category:
            subquery = sa.select(EvaluationRecord.session_id).join(ScenarioRecord).where(
                ScenarioRecord.category == category.value
            ).distinct()
            query = query.where(SessionRecord.id.in_(subquery))
        
        query = query.order_by(SessionRecord.overall_score.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())


class EvaluationRepository(BaseRepository):
    """Repository for evaluation operations."""
    
    async def create(self, evaluation: EvaluationResult, session_id: str) -> EvaluationRecord:
        """Create a new evaluation record."""
        record = EvaluationRecord(
            id=evaluation.id,
            task_id=evaluation.task_id,
            session_id=session_id,
            scenario_id=evaluation.scenario.id if evaluation.scenario else None,
            
            # Agent response
            agent_id=evaluation.response.agent_id,
            response_text=evaluation.response.response_text,
            confidence_level=evaluation.response.confidence_level,
            evidence_requests=evaluation.response.evidence_requests,
            red_flags_identified=evaluation.response.red_flags_identified,
            reasoning_steps=evaluation.response.reasoning_steps,
            response_time_ms=evaluation.response.response_time_ms,
            
            # Metrics
            skepticism_calibration=evaluation.metrics.skepticism_calibration,
            evidence_standard_score=evaluation.metrics.evidence_standard_score,
            red_flag_detection=evaluation.metrics.red_flag_detection,
            reasoning_quality=evaluation.metrics.reasoning_quality,
            overall_score=evaluation.metrics.overall_score,
            
            # Result metadata
            passed_evaluation=evaluation.passed_evaluation,
            analysis=evaluation.analysis,
            evaluation_notes=evaluation.evaluation_notes,
            evaluated_at=evaluation.evaluated_at
        )
        
        self.session.add(record)
        await self.session.flush()
        logger.debug(f"Created evaluation record: {evaluation.id}")
        return record
    
    async def get_by_id(self, evaluation_id: str) -> Optional[EvaluationRecord]:
        """Get evaluation by ID."""
        result = await self.session.execute(
            sa.select(EvaluationRecord)
            .options(selectinload(EvaluationRecord.scenario))
            .where(EvaluationRecord.id == evaluation_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_session(self, session_id: str) -> List[EvaluationRecord]:
        """Get all evaluations for a session."""
        result = await self.session.execute(
            sa.select(EvaluationRecord)
            .options(selectinload(EvaluationRecord.scenario))
            .where(EvaluationRecord.session_id == session_id)
            .order_by(EvaluationRecord.evaluated_at)
        )
        return list(result.scalars().all())
    
    async def get_by_scenario(self, scenario_id: str) -> List[EvaluationRecord]:
        """Get all evaluations for a scenario."""
        result = await self.session.execute(
            sa.select(EvaluationRecord)
            .where(EvaluationRecord.scenario_id == scenario_id)
            .order_by(EvaluationRecord.evaluated_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_performance_stats(self, 
                                  session_id: Optional[str] = None,
                                  scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        query = sa.select(
            sa.func.count(EvaluationRecord.id).label("total_evaluations"),
            sa.func.sum(sa.case((EvaluationRecord.passed_evaluation == True, 1), else_=0)).label("passed_evaluations"),
            sa.func.avg(EvaluationRecord.overall_score).label("avg_overall_score"),
            sa.func.avg(EvaluationRecord.skepticism_calibration).label("avg_skepticism"),
            sa.func.avg(EvaluationRecord.evidence_standard_score).label("avg_evidence"),
            sa.func.avg(EvaluationRecord.red_flag_detection).label("avg_red_flags"),
            sa.func.avg(EvaluationRecord.reasoning_quality).label("avg_reasoning"),
            sa.func.avg(EvaluationRecord.response_time_ms).label("avg_response_time")
        )
        
        if session_id:
            query = query.where(EvaluationRecord.session_id == session_id)
        if scenario_id:
            query = query.where(EvaluationRecord.scenario_id == scenario_id)
        
        result = await self.session.execute(query)
        row = result.first()
        
        if not row or row.total_evaluations == 0:
            return {
                "total_evaluations": 0,
                "passed_evaluations": 0,
                "pass_rate": 0.0,
                "avg_overall_score": 0.0,
                "avg_skepticism_calibration": 0.0,
                "avg_evidence_standard_score": 0.0,
                "avg_red_flag_detection": 0.0,
                "avg_reasoning_quality": 0.0,
                "avg_response_time_ms": 0.0
            }
        
        return {
            "total_evaluations": int(row.total_evaluations),
            "passed_evaluations": int(row.passed_evaluations or 0),
            "pass_rate": float(row.passed_evaluations or 0) / float(row.total_evaluations),
            "avg_overall_score": float(row.avg_overall_score or 0),
            "avg_skepticism_calibration": float(row.avg_skepticism or 0),
            "avg_evidence_standard_score": float(row.avg_evidence or 0),
            "avg_red_flag_detection": float(row.avg_red_flags or 0),
            "avg_reasoning_quality": float(row.avg_reasoning or 0),
            "avg_response_time_ms": float(row.avg_response_time or 0)
        }
    
    async def delete_by_session(self, session_id: str) -> int:
        """Delete all evaluations for a session."""
        result = await self.session.execute(
            sa.delete(EvaluationRecord).where(EvaluationRecord.session_id == session_id)
        )
        deleted_count = result.rowcount or 0
        await self.session.flush()
        logger.debug(f"Deleted {deleted_count} evaluation records for session: {session_id}")
        return deleted_count


class CacheRepository(BaseRepository):
    """Repository for caching operations."""
    
    def _generate_cache_id(self, key: str) -> str:
        """Generate cache ID from key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        cache_id = self._generate_cache_id(key)
        
        result = await self.session.execute(
            sa.select(CacheEntry).where(CacheEntry.id == cache_id)
        )
        entry = result.scalar_one_or_none()
        
        if not entry:
            return None
        
        if entry.is_expired:
            await self.delete(key)
            return None
        
        return entry.value
    
    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        cache_id = self._generate_cache_id(key)
        expires_at = None
        
        if ttl_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        # Try to update existing entry
        result = await self.session.execute(
            sa.select(CacheEntry).where(CacheEntry.id == cache_id)
        )
        entry = result.scalar_one_or_none()
        
        if entry:
            entry.value = value
            entry.expires_at = expires_at
        else:
            entry = CacheEntry(
                id=cache_id,
                key=key,
                value=value,
                expires_at=expires_at
            )
            self.session.add(entry)
        
        await self.session.flush()
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        cache_id = self._generate_cache_id(key)
        
        result = await self.session.execute(
            sa.delete(CacheEntry).where(CacheEntry.id == cache_id)
        )
        
        deleted = (result.rowcount or 0) > 0
        await self.session.flush()
        return deleted
    
    async def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        result = await self.session.execute(
            sa.delete(CacheEntry).where(
                CacheEntry.expires_at.is_not(None),
                CacheEntry.expires_at < datetime.utcnow()
            )
        )
        
        deleted_count = result.rowcount or 0
        await self.session.flush()
        logger.debug(f"Cleared {deleted_count} expired cache entries")
        return deleted_count


class AuditRepository(BaseRepository):
    """Repository for audit logging."""
    
    async def log(self, 
                 action: str,
                 resource_type: str,
                 resource_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 ip_address: Optional[str] = None,
                 user_agent: Optional[str] = None) -> AuditLog:
        """Create audit log entry."""
        log_entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.session.add(log_entry)
        await self.session.flush()
        return log_entry
    
    async def get_logs(self, 
                      user_id: Optional[str] = None,
                      action: Optional[str] = None,
                      resource_type: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> List[AuditLog]:
        """Get audit logs with filtering."""
        query = sa.select(AuditLog)
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        if action:
            query = query.where(AuditLog.action == action)
        if resource_type:
            query = query.where(AuditLog.resource_type == resource_type)
        
        query = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())