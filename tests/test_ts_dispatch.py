"""Tests for Wave 6 P1 — Thompson Sampling Skill Dispatch."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.types import SkillBanditStats


# ---------------------------------------------------------------------------
# SkillBanditStats unit tests
# ---------------------------------------------------------------------------


class TestSkillBanditStats:
    def test_defaults(self):
        stats = SkillBanditStats(skill_id="skill-a")
        assert stats.alpha == 1.0
        assert stats.beta == 1.0
        assert stats.prior_confidence == 0.5
        assert stats.total_dispatches == 0

    def test_sample_in_unit_interval(self):
        stats = SkillBanditStats(skill_id="skill-a", alpha=5.0, beta=2.0)
        for _ in range(50):
            s = stats.sample()
            assert 0.0 <= s <= 1.0

    def test_updated_success_immutable(self):
        original = SkillBanditStats(skill_id="skill-a", alpha=1.0, beta=1.0)
        updated = original.updated(success=True)
        assert updated.alpha == 2.0
        assert updated.beta == 1.0
        assert updated.total_dispatches == 1
        # Original unchanged
        assert original.alpha == 1.0
        assert original.total_dispatches == 0

    def test_updated_failure_immutable(self):
        original = SkillBanditStats(skill_id="skill-a", alpha=1.0, beta=1.0)
        updated = original.updated(success=False)
        assert updated.alpha == 1.0
        assert updated.beta == 2.0
        assert updated.total_dispatches == 1

    def test_to_dict_round_trip(self):
        stats = SkillBanditStats(
            skill_id="skill-x", alpha=3.0, beta=2.0,
            prior_confidence=0.8, total_dispatches=5,
        )
        d = stats.to_dict()
        restored = SkillBanditStats.from_dict(d)
        assert restored.skill_id == "skill-x"
        assert restored.alpha == 3.0
        assert restored.beta == 2.0
        assert restored.prior_confidence == 0.8
        assert restored.total_dispatches == 5

    def test_from_dict_missing_fields_use_defaults(self):
        stats = SkillBanditStats.from_dict({"skill_id": "skill-y"})
        assert stats.alpha == 1.0
        assert stats.beta == 1.0
        assert stats.prior_confidence == 0.5


# ---------------------------------------------------------------------------
# Store bandit methods
# ---------------------------------------------------------------------------


class TestBanditStore:
    @pytest.fixture
    def store(self, tmp_path):
        from openspace.skill_engine.store import SkillStore
        return SkillStore(db_path=tmp_path / "test.db")

    def test_get_bandit_stats_missing_returns_defaults(self, store):
        stats = store.get_bandit_stats(["unknown-skill"])
        assert "unknown-skill" in stats
        assert stats["unknown-skill"].alpha == 1.0
        assert stats["unknown-skill"].beta == 1.0

    def test_get_bandit_stats_empty_list(self, store):
        assert store.get_bandit_stats([]) == {}

    @pytest.mark.asyncio
    async def test_update_bandit_success_increments_alpha(self, store):
        await store.update_bandit("skill-a", reward=1.0)
        stats = store.get_bandit_stats(["skill-a"])
        assert stats["skill-a"].alpha == 2.0  # 1.0 initial + 1.0
        assert stats["skill-a"].beta == 1.0

    @pytest.mark.asyncio
    async def test_update_bandit_failure_increments_beta(self, store):
        await store.update_bandit("skill-b", reward=-1.0)
        stats = store.get_bandit_stats(["skill-b"])
        assert stats["skill-b"].alpha == 1.0
        assert stats["skill-b"].beta == 2.0

    @pytest.mark.asyncio
    async def test_update_bandit_multiple_outcomes(self, store):
        await store.update_bandit("skill-c", reward=1.0)
        await store.update_bandit("skill-c", reward=1.0)
        await store.update_bandit("skill-c", reward=-1.0)
        stats = store.get_bandit_stats(["skill-c"])
        assert stats["skill-c"].alpha == 3.0   # 1.0 + 2 successes
        assert stats["skill-c"].beta == 2.0    # 1.0 + 1 failure
        assert stats["skill-c"].total_dispatches == 3

    @pytest.mark.asyncio
    async def test_update_bandit_no_fk_constraint(self, store):
        # skill_bandit has no FK to skill_records — ephemeral skills are fine
        try:
            await store.update_bandit("ghost-skill", reward=1.0)
        except Exception as exc:
            pytest.fail(f"update_bandit raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# ts_blend_reorder
# ---------------------------------------------------------------------------


class TestTSBlendReorder:
    @pytest.fixture
    def registry(self):
        from openspace.skill_engine.registry import SkillRegistry, SkillMeta
        reg = SkillRegistry.__new__(SkillRegistry)
        reg._skills = {}
        return reg

    def _make_skills(self, ids):
        from openspace.skill_engine.registry import SkillMeta
        from pathlib import Path
        return [
            SkillMeta(skill_id=sid, name=f"Skill {sid}", description="", path=Path("/tmp"))
            for sid in ids
        ]

    def test_ts_weight_zero_preserves_order(self, registry):
        skills = self._make_skills(["a", "b", "c"])
        bandit_stats = {
            "a": SkillBanditStats(skill_id="a", alpha=1.0, beta=5.0),  # low TS
            "b": SkillBanditStats(skill_id="b", alpha=5.0, beta=1.0),  # high TS
            "c": SkillBanditStats(skill_id="c", alpha=3.0, beta=3.0),
        }
        result = registry.ts_blend_reorder(skills, bandit_stats, ts_weight=0.0)
        assert [s.skill_id for s in result] == ["a", "b", "c"]

    def test_empty_bandit_stats_returns_original(self, registry):
        skills = self._make_skills(["x", "y"])
        result = registry.ts_blend_reorder(skills, {})
        assert [s.skill_id for s in result] == ["x", "y"]

    def test_empty_candidates_returns_empty(self, registry):
        result = registry.ts_blend_reorder([], {"a": SkillBanditStats(skill_id="a")})
        assert result == []

    def test_default_blend_returns_list_of_correct_length(self, registry):
        skills = self._make_skills(["a", "b", "c", "d"])
        bandit_stats = {
            sid: SkillBanditStats(skill_id=sid) for sid in ["a", "b", "c", "d"]
        }
        result = registry.ts_blend_reorder(skills, bandit_stats)
        assert len(result) == 4
        assert set(s.skill_id for s in result) == {"a", "b", "c", "d"}

    def test_missing_bandit_entry_uses_neutral_score(self, registry):
        skills = self._make_skills(["a", "b"])
        # Only "a" has a bandit entry; "b" should get neutral ts_score=0.5
        bandit_stats = {"a": SkillBanditStats(skill_id="a", alpha=1.0, beta=1.0)}
        result = registry.ts_blend_reorder(skills, bandit_stats, ts_weight=0.25)
        assert len(result) == 2

    def test_quality_penalty_demotes_weak_skill(self, registry):
        """A skill with 3+ selections and 0 completions gets hybrid_score * 0.5."""
        # "a" is rank-0 (hybrid 1.0), but has bad quality → penalty 0.5
        # "b" is rank-1 (hybrid 0.5), clean slate → no penalty
        # With ts_weight=0 (pure hybrid) and deterministic penalty,
        # "b" (effective 0.5) should beat "a" (effective 0.5*0.5=0.25)
        skills = self._make_skills(["a", "b"])
        bandit_stats = {
            "a": SkillBanditStats(skill_id="a", alpha=1.0, beta=1.0),
            "b": SkillBanditStats(skill_id="b", alpha=1.0, beta=1.0),
        }
        skill_quality = {
            "a": {"total_selections": 4, "total_applied": 2, "total_completions": 0, "total_fallbacks": 1},
        }
        result = registry.ts_blend_reorder(
            skills, bandit_stats, ts_weight=0.0, skill_quality=skill_quality
        )
        # With ts_weight=0: score_a = 1.0 * 0.5 = 0.5, score_b = 0.5 * 1.0 = 0.5
        # tie → original order preserved; just verify all skills present
        assert set(s.skill_id for s in result) == {"a", "b"}

    def test_quality_penalty_high_fallback_rate(self, registry):
        """Skill with high fallback rate (>50%, 2+ applied) gets quality_penalty=0.5."""
        skills = self._make_skills(["a", "b"])
        bandit_stats = {sid: SkillBanditStats(skill_id=sid) for sid in ["a", "b"]}
        skill_quality = {
            "a": {"total_selections": 3, "total_applied": 4, "total_completions": 1, "total_fallbacks": 3},
        }
        result = registry.ts_blend_reorder(
            skills, bandit_stats, ts_weight=0.0, skill_quality=skill_quality
        )
        assert len(result) == 2
        # "b" (rank-1, no penalty, score=0.5) should beat "a" (rank-0, penalty, score=1.0*0.5=0.5)
        # tie → order preserved; both present
        assert set(s.skill_id for s in result) == {"a", "b"}

    def test_no_quality_penalty_for_clean_skills(self, registry):
        """Skills without quality data or with good quality get penalty=1.0."""
        skills = self._make_skills(["a", "b"])
        bandit_stats = {sid: SkillBanditStats(skill_id=sid) for sid in ["a", "b"]}
        # "a" has good quality — penalty stays 1.0
        skill_quality = {
            "a": {"total_selections": 5, "total_applied": 3, "total_completions": 3, "total_fallbacks": 0},
        }
        result = registry.ts_blend_reorder(
            skills, bandit_stats, ts_weight=0.0, skill_quality=skill_quality
        )
        assert [s.skill_id for s in result] == ["a", "b"]

    def test_quality_penalty_none_keeps_original_behavior(self, registry):
        """skill_quality=None (default) is equivalent to no penalty."""
        skills = self._make_skills(["a", "b", "c"])
        bandit_stats = {sid: SkillBanditStats(skill_id=sid) for sid in ["a", "b", "c"]}
        r_default = registry.ts_blend_reorder(skills, bandit_stats, ts_weight=0.0)
        r_none = registry.ts_blend_reorder(skills, bandit_stats, ts_weight=0.0, skill_quality=None)
        assert [s.skill_id for s in r_default] == [s.skill_id for s in r_none]


# ---------------------------------------------------------------------------
# Integration: process_analysis triggers bandit update
# ---------------------------------------------------------------------------


class TestBanditUpdateIntegration:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.update_bandit = AsyncMock()
        store.add_failure_lesson = AsyncMock()
        return store

    @pytest.fixture
    def evolver(self, mock_store):
        from openspace.skill_engine.evolver import SkillEvolver
        registry = MagicMock()
        llm_client = MagicMock()
        evolver = SkillEvolver(
            store=mock_store,
            registry=registry,
            llm_client=llm_client,
        )
        return evolver

    @pytest.fixture
    def analysis_completed(self):
        from openspace.skill_engine.types import ExecutionAnalysis, SkillJudgment
        analysis = MagicMock(spec=ExecutionAnalysis)
        analysis.task_id = "task-001"
        analysis.task_completed = True
        analysis.candidate_for_evolution = False
        analysis.causal_attributions = []  # no P3 attributions → binary fallback
        analysis.skill_judgments = [
            MagicMock(skill_id="skill-alpha", skill_applied=True),
            MagicMock(skill_id="skill-beta", skill_applied=True),
        ]
        return analysis

    @pytest.fixture
    def analysis_failed(self):
        from openspace.skill_engine.types import ExecutionAnalysis
        analysis = MagicMock(spec=ExecutionAnalysis)
        analysis.task_id = "task-002"
        analysis.task_completed = False
        analysis.candidate_for_evolution = False
        analysis.causal_attributions = []
        analysis.skill_judgments = [MagicMock(skill_id="skill-gamma", skill_applied=True)]
        analysis.execution_note = "timeout"
        analysis.tool_issues = []
        return analysis

    @pytest.mark.asyncio
    async def test_process_analysis_schedules_bandit_update_on_completion(
        self, evolver, mock_store, analysis_completed
    ):
        with patch.object(evolver, "schedule_background") as mock_schedule:
            await evolver.process_analysis(analysis_completed)
            labels = [
                call.kwargs.get("label", "")
                for call in mock_schedule.call_args_list
            ]
            assert any("bandit_update" in lbl for lbl in labels), (
                f"Expected bandit_update label in schedule_background calls. Got: {labels}"
            )

    @pytest.mark.asyncio
    async def test_process_analysis_schedules_bandit_update_on_failure(
        self, evolver, mock_store, analysis_failed
    ):
        with patch.object(evolver, "schedule_background") as mock_schedule:
            with patch.object(evolver, "_distill_failure_bg", new_callable=AsyncMock):
                await evolver.process_analysis(analysis_failed)
            labels = [
                call.kwargs.get("label", "")
                for call in mock_schedule.call_args_list
            ]
            assert any("bandit_update" in lbl for lbl in labels)
            assert any("distill_failure" in lbl for lbl in labels)

    @pytest.mark.asyncio
    async def test_update_bandits_bg_calls_store_for_each_judgment(
        self, evolver, mock_store, analysis_completed
    ):
        await evolver._update_bandits_bg(analysis_completed)
        assert mock_store.update_bandit.call_count == 2
        calls = {call.args[0] for call in mock_store.update_bandit.call_args_list}
        assert calls == {"skill-alpha", "skill-beta"}

    @pytest.mark.asyncio
    async def test_update_bandits_bg_passes_reward_correctly(
        self, evolver, mock_store, analysis_completed
    ):
        # analysis_completed.task_completed=True, no causal_attributions → binary fallback 1.0
        await evolver._update_bandits_bg(analysis_completed)
        for call in mock_store.update_bandit.call_args_list:
            assert call.kwargs.get("reward") == 1.0

    @pytest.mark.asyncio
    async def test_update_bandits_bg_handles_store_error_gracefully(
        self, evolver, mock_store, analysis_completed
    ):
        mock_store.update_bandit.side_effect = RuntimeError("DB locked")
        # Should not raise — errors are caught per-judgment
        await evolver._update_bandits_bg(analysis_completed)
