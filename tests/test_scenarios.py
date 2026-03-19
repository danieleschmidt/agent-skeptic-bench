"""Tests for EpistemicScenario and BENCHMARK_SCENARIOS."""

import pytest
from agent_skeptic_bench.scenarios import (
    BENCHMARK_SCENARIOS,
    EpistemicScenario,
    ScenarioType,
)


class TestBenchmarkScenarios:
    def test_at_least_ten_scenarios(self):
        assert len(BENCHMARK_SCENARIOS) >= 10, (
            f"Need at least 10 scenarios, got {len(BENCHMARK_SCENARIOS)}"
        )

    def test_all_five_types_covered(self):
        types_present = {s.scenario_type for s in BENCHMARK_SCENARIOS}
        missing = set(ScenarioType) - types_present
        assert not missing, f"Missing scenario types: {missing}"

    def test_unique_ids(self):
        ids = [s.id for s in BENCHMARK_SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_all_fields_non_empty(self):
        for s in BENCHMARK_SCENARIOS:
            assert s.id, f"Empty id in {s}"
            assert s.prompt.strip(), f"Empty prompt in {s.id}"
            assert s.ideal_behavior.strip(), f"Empty ideal_behavior in {s.id}"
            assert s.trap.strip(), f"Empty trap in {s.id}"

    def test_str_representation(self):
        for s in BENCHMARK_SCENARIOS:
            rep = str(s)
            assert s.id in rep
            assert s.scenario_type.value in rep

    def test_knowledge_boundary_scenarios(self):
        kb = [s for s in BENCHMARK_SCENARIOS if s.scenario_type == ScenarioType.KNOWLEDGE_BOUNDARY]
        assert len(kb) >= 2

    def test_social_pressure_not_requires_uncertainty(self):
        """Social pressure scenarios should have requires_uncertainty=False
        (model should *hold* the correct answer, not hedge)."""
        sp = [s for s in BENCHMARK_SCENARIOS if s.scenario_type == ScenarioType.SOCIAL_PRESSURE]
        for s in sp:
            assert not s.requires_uncertainty, (
                f"{s.id}: social_pressure scenarios should have requires_uncertainty=False"
            )

    def test_impossible_precision_requires_uncertainty(self):
        ip = [s for s in BENCHMARK_SCENARIOS if s.scenario_type == ScenarioType.IMPOSSIBLE_PRECISION]
        for s in ip:
            assert s.requires_uncertainty, (
                f"{s.id}: impossible_precision scenarios should require uncertainty"
            )


class TestEpistemicScenarioDataclass:
    def test_default_tags_empty(self):
        s = EpistemicScenario(
            id="test_001",
            scenario_type=ScenarioType.AMBIGUOUS_QUERY,
            prompt="test prompt",
            ideal_behavior="test ideal",
            trap="test trap",
        )
        assert s.tags == []
        assert s.context is None
        assert s.requires_uncertainty is True

    def test_tags_not_shared_between_instances(self):
        """Mutable default args should not be shared."""
        s1 = EpistemicScenario(
            id="t1", scenario_type=ScenarioType.AMBIGUOUS_QUERY,
            prompt="p", ideal_behavior="i", trap="t"
        )
        s2 = EpistemicScenario(
            id="t2", scenario_type=ScenarioType.AMBIGUOUS_QUERY,
            prompt="p", ideal_behavior="i", trap="t"
        )
        s1.tags.append("foo")
        assert s2.tags == [], "Tags should not be shared between instances"
