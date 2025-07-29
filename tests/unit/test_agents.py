"""Unit tests for skeptic agent creation and functionality."""

import pytest
from unittest.mock import Mock, patch
from agent_skeptic_bench.agents import create_skeptic_agent


class TestSkepticAgents:
    """Test suite for skeptic agent functionality."""

    def test_create_skeptic_agent_basic(self, mock_llm_client):
        """Test basic skeptic agent creation."""
        agent = create_skeptic_agent(
            model="test-model",
            client=mock_llm_client
        )
        assert agent is not None

    def test_skepticism_level_validation(self, mock_llm_client):
        """Test that skepticism level parameter is validated."""
        valid_levels = ["low", "calibrated", "high"]
        
        for level in valid_levels:
            agent = create_skeptic_agent(
                model="test-model",
                client=mock_llm_client,
                skepticism_level=level
            )
            assert agent is not None

    def test_evidence_standards_validation(self, mock_llm_client):
        """Test that evidence standards parameter is validated."""
        valid_standards = ["minimal", "scientific", "rigorous"]
        
        for standard in valid_standards:
            agent = create_skeptic_agent(
                model="test-model",
                client=mock_llm_client,
                evidence_standards=standard
            )
            assert agent is not None

    def test_agent_evaluate_claim(self, sample_skeptic_agent, sample_false_claim):
        """Test agent's ability to evaluate false claims."""
        # This would test the actual evaluation logic once implemented
        pass