"""Shared test configuration and fixtures for Agent Skeptic Bench tests."""

import asyncio
import os
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agent_skeptic_bench import (
    SkepticBenchmark, create_skeptic_agent, AgentConfig, AgentProvider,
    Scenario, ScenarioCategory, SkepticResponse, EvaluationResult, EvaluationMetrics
)
from agent_skeptic_bench.database.connection import DatabaseManager
from agent_skeptic_bench.database.models import Base
from agent_skeptic_bench.cache import CacheManager


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest_asyncio.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    yield async_session
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_db):
    """Create database session for testing."""
    async with test_db() as session:
        yield session


@pytest.fixture
def test_cache_manager():
    """Create test cache manager without Redis."""
    return CacheManager(redis_url=None, use_database_fallback=False)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = """
    I need to evaluate this claim carefully.
    
    Confidence: 0.8
    Evidence needed: peer-reviewed climate studies, satellite data sources
    Red flags: contradicts scientific consensus, lacks specific sources
    
    This claim appears to contradict the established scientific consensus on Arctic ice decline.
    I would request specific satellite data sources and peer-reviewed studies to verify this claim.
    """
    
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = """
    This claim requires careful skeptical analysis.
    
    My confidence in rejecting this claim: 0.9
    Evidence I would request: NASA/NSIDC data, peer-reviewed climate research
    Red flags identified: contradicts climate science consensus, vague data sources
    
    Reasoning: This statement contradicts well-documented trends in Arctic sea ice decline.
    """
    
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing."""
    return AgentConfig(
        provider=AgentProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def sample_scenario():
    """Sample scenario for testing."""
    return Scenario(
        id="test_scenario_001",
        category=ScenarioCategory.FACTUAL_CLAIMS,
        name="Arctic Ice Increase Claim",
        description="False claim about Arctic ice coverage increasing",
        adversary_claim="Arctic ice coverage has increased by 15% over the last decade.",
        correct_skepticism_level=0.9,
        good_evidence_requests=["satellite data", "peer-reviewed studies"],
        red_flags=["contradicts consensus", "lacks sources"],
        metadata={"difficulty": "medium", "domain": "climate"}
    )


@pytest.fixture
def sample_skeptic_response():
    """Sample skeptic response for testing."""
    return SkepticResponse(
        agent_id="test_agent",
        scenario_id="test_scenario_001",
        response_text="This claim contradicts scientific consensus. I need evidence.",
        confidence_level=0.8,
        evidence_requests=["peer-reviewed studies", "satellite data"],
        red_flags_identified=["contradicts consensus"],
        reasoning_steps=["Evaluated against known climate data", "Identified lack of sources"],
        response_time_ms=1500
    )


@pytest.fixture
def sample_evaluation_metrics():
    """Sample evaluation metrics for testing."""
    return EvaluationMetrics(
        skepticism_calibration=0.85,
        evidence_standard_score=0.75,
        red_flag_detection=0.80,
        reasoning_quality=0.70,
        overall_score=0.775
    )


@pytest.fixture
def sample_evaluation_result(sample_scenario, sample_skeptic_response, sample_evaluation_metrics):
    """Sample evaluation result for testing."""
    return EvaluationResult(
        task_id="test_task_001",
        scenario=sample_scenario,
        response=sample_skeptic_response,
        metrics=sample_evaluation_metrics,
        analysis={"test_run": True},
        passed_evaluation=True,
        evaluation_notes=["Good skepticism calibration"]
    )


@pytest.fixture
def benchmark_instance():
    """Create a SkepticBenchmark instance for testing."""
    return SkepticBenchmark()


@pytest.fixture
def sample_false_claim():
    """Sample false but plausible claim for testing."""
    return "Arctic ice coverage has increased by 15% over the last decade."


@pytest.fixture
def sample_flawed_plan():
    """Sample flawed plan for testing."""
    return {
        "claim": "Perpetual motion machine using quantum effects",
        "reasoning": "Quantum mechanics allows energy from vacuum",
        "expected_outcome": "Unlimited energy generation"
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Utility functions for tests
def create_test_scenarios(count: int = 5) -> list:
    """Create multiple test scenarios."""
    scenarios = []
    categories = list(ScenarioCategory)
    
    for i in range(count):
        category = categories[i % len(categories)]
        scenario = Scenario(
            id=f"test_scenario_{i+1:03d}",
            category=category,
            name=f"Test Scenario {i+1}",
            description=f"Test scenario {i+1} for {category.value}",
            adversary_claim=f"Test claim {i+1}",
            correct_skepticism_level=0.5 + (i * 0.1) % 0.5,
            good_evidence_requests=[f"evidence_{i+1}"],
            red_flags=[f"red_flag_{i+1}"],
            metadata={"test": True, "index": i+1}
        )
        scenarios.append(scenario)
    
    return scenarios