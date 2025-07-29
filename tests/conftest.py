"""Shared test configuration and fixtures for Agent Skeptic Bench tests."""

import pytest
from unittest.mock import Mock
from agent_skeptic_bench import SkepticBenchmark, create_skeptic_agent


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing without API calls."""
    client = Mock()
    client.chat.completions.create.return_value.choices[0].message.content = "Mock response"
    return client


@pytest.fixture
def sample_skeptic_agent(mock_llm_client):
    """Create a test skeptic agent with mocked dependencies."""
    return create_skeptic_agent(
        model="test-model",
        client=mock_llm_client,
        skepticism_level="calibrated"
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