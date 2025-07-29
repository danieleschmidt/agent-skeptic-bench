"""Integration tests for full benchmark evaluation."""

import pytest
from unittest.mock import Mock, patch
from agent_skeptic_bench import run_full_evaluation


class TestFullEvaluation:
    """Test suite for full evaluation integration."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_run_full_evaluation_basic(self, sample_skeptic_agent):
        """Test basic full evaluation run."""
        # Mock the evaluation to avoid actual API calls
        with patch('agent_skeptic_bench.evaluation.evaluate_scenario') as mock_eval:
            mock_eval.return_value = Mock(
                appropriateness_score=0.85,
                evidence_standard="scientific",
                final_belief="skeptical"
            )
            
            results = run_full_evaluation(
                skeptic_agent=sample_skeptic_agent,
                categories=["factual_claims"],
                parallel=False
            )
            
            assert results is not None

    @pytest.mark.integration
    def test_evaluation_with_multiple_categories(self, sample_skeptic_agent):
        """Test evaluation across multiple scenario categories."""
        categories = ["factual_claims", "flawed_plans"]
        
        with patch('agent_skeptic_bench.evaluation.evaluate_scenario') as mock_eval:
            mock_eval.return_value = Mock(
                appropriateness_score=0.8,
                evidence_standard="peer_reviewed",
                final_belief="highly_skeptical"
            )
            
            results = run_full_evaluation(
                skeptic_agent=sample_skeptic_agent,
                categories=categories,
                parallel=True
            )
            
            assert results is not None

    @pytest.mark.integration
    def test_results_serialization(self, sample_skeptic_agent, tmp_path):
        """Test that evaluation results can be saved and loaded."""
        output_file = tmp_path / "test_results.json"
        
        with patch('agent_skeptic_bench.evaluation.evaluate_scenario') as mock_eval:
            mock_eval.return_value = Mock(
                appropriateness_score=0.9,
                evidence_standard="rigorous",
                final_belief="appropriately_skeptical"
            )
            
            results = run_full_evaluation(
                skeptic_agent=sample_skeptic_agent,
                categories=["factual_claims"],
                save_results=str(output_file)
            )
            
            assert output_file.exists()
            assert results is not None