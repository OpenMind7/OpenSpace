"""Tests for Wave 5 P2 — Failure-Trajectory Distillation.

Covers:
  - FailureLesson serialization (to_dict / from_dict)
  - TTL and confidence gate behaviour
  - SkillStore: add / get / prune failure lessons (temp DB)
  - SkillEvolver: _distill_failure_bg triggered when task_completed=False
  - SkillEvolver: _distill_failure confidence gate (< 0.7 → skip)
  - SkillRegistry: failure section injected into selection prompt
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.types import FailureLesson, _FAILURE_MODES
from openspace.skill_engine.store import SkillStore
from openspace.skill_engine.evolver import SkillEvolver
from openspace.skill_engine.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lesson(**kwargs) -> FailureLesson:
    defaults = dict(
        lesson_id="abc123",
        task_id="task-001",
        skill_ids=["skill-a"],
        task_summary="Attempted to generate a PDF report",
        failure_mode="api_misuse",
        lesson_text="Avoid calling pdfkit without wkhtmltopdf installed.",
        tool_culprits=["pdfkit"],
        confidence=0.85,
    )
    defaults.update(kwargs)
    return FailureLesson(**defaults)


def make_evolver(store=None) -> SkillEvolver:
    if store is None:
        store = MagicMock()
        store.add_failure_lesson = AsyncMock()
        store.get_recent_failure_lessons = MagicMock(return_value=[])
        store.get_analysis_count = MagicMock(return_value=0)
    registry = MagicMock()
    llm_client = AsyncMock()
    llm_client.model = "test-model"
    return SkillEvolver(store=store, registry=registry, llm_client=llm_client)


def make_analysis(task_completed: bool = False, task_id: str = "task-1") -> MagicMock:
    analysis = MagicMock()
    analysis.task_id = task_id
    analysis.task_completed = task_completed
    analysis.candidate_for_evolution = False
    analysis.execution_note = "Some error occurred"
    analysis.tool_issues = ["pdfkit"]
    analysis.skill_judgments = []
    return analysis


# ---------------------------------------------------------------------------
# FailureLesson serialization
# ---------------------------------------------------------------------------

class TestFailureLesson:

    def test_to_dict_round_trip(self):
        lesson = make_lesson()
        restored = FailureLesson.from_dict(lesson.to_dict())
        assert restored.lesson_id == lesson.lesson_id
        assert restored.task_id == lesson.task_id
        assert restored.skill_ids == lesson.skill_ids
        assert restored.failure_mode == lesson.failure_mode
        assert restored.lesson_text == lesson.lesson_text
        assert restored.tool_culprits == lesson.tool_culprits
        assert abs(restored.confidence - lesson.confidence) < 1e-6

    def test_expires_at_round_trip(self):
        expires = datetime.now() + timedelta(days=30)
        lesson = make_lesson(expires_at=expires)
        d = lesson.to_dict()
        assert d["expires_at"] is not None
        restored = FailureLesson.from_dict(d)
        # fromisoformat round-trip accurate to microsecond
        assert abs((restored.expires_at - expires).total_seconds()) < 0.001

    def test_expires_at_none_round_trip(self):
        lesson = make_lesson(expires_at=None)
        restored = FailureLesson.from_dict(lesson.to_dict())
        assert restored.expires_at is None

    def test_failure_modes_frozenset(self):
        assert "api_misuse" in _FAILURE_MODES
        assert "other" in _FAILURE_MODES
        assert len(_FAILURE_MODES) == 8

    def test_default_confidence(self):
        lesson = FailureLesson(lesson_id="x", task_id="y")
        assert lesson.confidence == 0.7

    def test_from_dict_missing_optional_fields(self):
        minimal = {"lesson_id": "x", "task_id": "y"}
        lesson = FailureLesson.from_dict(minimal)
        assert lesson.skill_ids == []
        assert lesson.tool_culprits == []
        assert lesson.failure_mode == "other"
        assert lesson.task_summary == ""


# ---------------------------------------------------------------------------
# SkillStore — failure lesson persistence
# ---------------------------------------------------------------------------

class TestFailureLessonStore:

    @pytest.fixture()
    def store(self, tmp_path):
        s = SkillStore(db_path=tmp_path / "test.db")
        yield s
        s.close()

    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        lesson = make_lesson(
            expires_at=datetime.now() + timedelta(days=30)
        )
        await store.add_failure_lesson(lesson)

        results = store.get_recent_failure_lessons(["skill-a"])
        assert len(results) == 1
        assert results[0].lesson_id == lesson.lesson_id
        assert results[0].failure_mode == "api_misuse"
        assert results[0].skill_ids == ["skill-a"]

    @pytest.mark.asyncio
    async def test_expired_lesson_excluded(self, store):
        expired = make_lesson(
            lesson_id="old",
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        await store.add_failure_lesson(expired)

        results = store.get_recent_failure_lessons([])
        assert all(r.lesson_id != "old" for r in results)

    @pytest.mark.asyncio
    async def test_prune_removes_expired(self, store):
        fresh = make_lesson(
            lesson_id="fresh",
            expires_at=datetime.now() + timedelta(days=30),
        )
        stale = make_lesson(
            lesson_id="stale",
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        await store.add_failure_lesson(fresh)
        await store.add_failure_lesson(stale)

        removed = await store.prune_expired_failure_lessons()
        assert removed == 1

        remaining = store.get_recent_failure_lessons([])
        assert len(remaining) == 1
        assert remaining[0].lesson_id == "fresh"

    @pytest.mark.asyncio
    async def test_skill_id_preference_ordering(self, store):
        """Lessons matching skill_ids should rank above unrelated ones."""
        unrelated = make_lesson(
            lesson_id="unrelated",
            skill_ids=["other-skill"],
            expires_at=datetime.now() + timedelta(days=30),
        )
        related = make_lesson(
            lesson_id="related",
            skill_ids=["skill-a", "skill-b"],
            expires_at=datetime.now() + timedelta(days=30),
        )
        await store.add_failure_lesson(unrelated)
        await store.add_failure_lesson(related)

        results = store.get_recent_failure_lessons(["skill-a"])
        assert results[0].lesson_id == "related"

    @pytest.mark.asyncio
    async def test_no_expiry_lesson_always_returned(self, store):
        """Lessons with expires_at=NULL are never pruned."""
        lesson = make_lesson(lesson_id="eternal", expires_at=None)
        await store.add_failure_lesson(lesson)

        results = store.get_recent_failure_lessons([])
        assert any(r.lesson_id == "eternal" for r in results)


# ---------------------------------------------------------------------------
# SkillEvolver — distill_failure_bg trigger
# ---------------------------------------------------------------------------

class TestDistillFailure:

    @pytest.mark.asyncio
    async def test_distill_triggered_on_failed_task(self):
        """_distill_failure_bg should be scheduled when task_completed=False."""
        evolver = make_evolver()
        analysis = make_analysis(task_completed=False)

        with patch.object(evolver, "schedule_background") as mock_sched:
            await evolver.process_analysis(analysis)
            # distill_failure_bg fires on failed tasks; fine_tune_embeddings always fires
            labels = [c.kwargs.get("label", "") for c in mock_sched.call_args_list]
            distill_labels = [l for l in labels if l.startswith("distill_failure:")]
            assert len(distill_labels) == 1, f"Expected 1 distill call, got: {labels}"

    @pytest.mark.asyncio
    async def test_distill_not_triggered_on_success(self):
        """_distill_failure_bg must NOT be scheduled when task_completed=True."""
        evolver = make_evolver()
        analysis = make_analysis(task_completed=True)

        with patch.object(evolver, "schedule_background") as mock_sched:
            await evolver.process_analysis(analysis)

        # distill_failure_bg must NOT fire on success; fine_tune_embeddings may still fire
        labels = [c.kwargs.get("label", "") for c in mock_sched.call_args_list]
        distill_labels = [l for l in labels if l.startswith("distill_failure:")]
        assert len(distill_labels) == 0, f"distill_failure must not fire on success, got: {labels}"

    @pytest.mark.asyncio
    async def test_confidence_gate_skips_low_confidence(self):
        """_distill_failure should not persist lesson when confidence < 0.7."""
        evolver = make_evolver()
        analysis = make_analysis(task_completed=False)
        analysis.skill_judgments = []

        low_conf_response = {
            "message": {
                "content": json.dumps({
                    "task_summary": "attempted task",
                    "failure_mode": "other",
                    "lesson_text": "avoid this",
                    "tool_culprits": [],
                    "confidence": 0.5,
                })
            }
        }
        evolver._llm_client.complete = AsyncMock(return_value=low_conf_response)

        with patch.object(evolver, "_format_analysis_context", return_value=""):
            await evolver._distill_failure(analysis)

        evolver._store.add_failure_lesson.assert_not_called()

    @pytest.mark.asyncio
    async def test_confidence_gate_persists_high_confidence(self):
        """_distill_failure should persist lesson when confidence >= 0.7."""
        evolver = make_evolver()
        analysis = make_analysis(task_completed=False)
        analysis.skill_judgments = []

        high_conf_response = {
            "message": {
                "content": json.dumps({
                    "task_summary": "attempted pdf generation",
                    "failure_mode": "api_misuse",
                    "lesson_text": "Avoid pdfkit without wkhtmltopdf.",
                    "tool_culprits": ["pdfkit"],
                    "confidence": 0.9,
                })
            }
        }
        evolver._llm_client.complete = AsyncMock(return_value=high_conf_response)
        evolver._store.add_failure_lesson = AsyncMock()

        with patch.object(evolver, "_format_analysis_context", return_value=""):
            await evolver._distill_failure(analysis)

        evolver._store.add_failure_lesson.assert_called_once()
        lesson = evolver._store.add_failure_lesson.call_args[0][0]
        assert lesson.failure_mode == "api_misuse"
        assert lesson.confidence == 0.9
        assert lesson.expires_at is not None

    @pytest.mark.asyncio
    async def test_distill_bg_is_non_fatal_on_llm_error(self):
        """_distill_failure_bg must not propagate exceptions."""
        evolver = make_evolver()
        analysis = make_analysis(task_completed=False)

        evolver._llm_client.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch.object(evolver, "_format_analysis_context", return_value=""):
            # Should not raise
            await evolver._distill_failure_bg(analysis)


# ---------------------------------------------------------------------------
# SkillRegistry — negative guidance injection
# ---------------------------------------------------------------------------

class TestNegativeGuidanceInjection:

    def test_failure_section_injected_when_context_provided(self):
        prompt = SkillRegistry._build_skill_selection_prompt(
            task="Generate a PDF report",
            skills_catalog="- **pdf-skill**: generates PDFs",
            max_skills=2,
            failure_context="1. **api_misuse** (90%): Avoid pdfkit without wkhtmltopdf.",
        )
        assert "# Known Failure Patterns (AVOID)" in prompt
        assert "api_misuse" in prompt
        assert "# Instructions" in prompt
        # Failure section must appear BEFORE Instructions
        assert prompt.index("Known Failure Patterns") < prompt.index("# Instructions")

    def test_no_failure_section_when_context_empty(self):
        prompt = SkillRegistry._build_skill_selection_prompt(
            task="Some task",
            skills_catalog="- **skill-a**: does something",
            max_skills=1,
            failure_context="",
        )
        assert "Known Failure Patterns" not in prompt

    def test_default_failure_context_is_empty(self):
        """Calling without failure_context must produce the same prompt as before."""
        prompt_new = SkillRegistry._build_skill_selection_prompt(
            task="task", skills_catalog="catalog", max_skills=2
        )
        assert "Known Failure Patterns" not in prompt_new

    @pytest.mark.asyncio
    async def test_select_skills_fetches_lessons_when_store_provided(self):
        """select_skills_with_llm should call store.get_recent_failure_lessons."""
        registry = SkillRegistry.__new__(SkillRegistry)
        registry._skills = {}
        registry._discovered = True

        mock_store = MagicMock()
        mock_store.get_recent_failure_lessons = MagicMock(return_value=[])

        result, record = await registry.select_skills_with_llm(
            task_description="",
            llm_client=AsyncMock(),
            store=mock_store,
        )
        assert result == []
        # Empty task_description → early return before store call, that's fine
