"""Tests for Wave 6 MEDIUM-5 — skill_dispatch_events durable audit trail.

Covers:
  - SkillStore: record_dispatch_event writes a row (temp DB)
  - SkillStore: multiple events for same task_id are all stored
  - SkillStore: clear() removes dispatch events too
  - tool_layer: _select_and_inject_skills writes dispatch event when store present
  - tool_layer: no dispatch event written when store is None
  - tool_layer: dispatch event is non-fatal (store failure doesn't propagate)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.store import SkillStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_store() -> SkillStore:
    tmp = tempfile.mkdtemp()
    return SkillStore(db_path=Path(tmp) / "test_dispatch.db")


# ---------------------------------------------------------------------------
# SkillStore — dispatch event persistence
# ---------------------------------------------------------------------------

class TestDispatchEventStore:
    @pytest.mark.asyncio
    async def test_record_single_event(self):
        store = make_temp_store()
        try:
            await store.record_dispatch_event(
                task_id="task-001",
                skill_ids=["skill-a", "skill-b"],
                method="llm",
            )
            with store._reader() as conn:
                rows = conn.execute(
                    "SELECT * FROM skill_dispatch_events WHERE task_id = ?",
                    ("task-001",),
                ).fetchall()
            assert len(rows) == 1
            assert json.loads(rows[0]["skill_ids"]) == ["skill-a", "skill-b"]
            assert rows[0]["method"] == "llm"
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_record_multiple_events_same_task(self):
        """Multiple dispatch events for the same task_id are all stored (no UNIQUE)."""
        store = make_temp_store()
        try:
            for i in range(3):
                await store.record_dispatch_event(
                    task_id="task-multi",
                    skill_ids=[f"skill-{i}"],
                    method="ts_blend",
                )
            with store._reader() as conn:
                rows = conn.execute(
                    "SELECT * FROM skill_dispatch_events WHERE task_id = ?",
                    ("task-multi",),
                ).fetchall()
            assert len(rows) == 3
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_record_empty_skill_ids(self):
        """Empty skill_ids list is stored as '[]'."""
        store = make_temp_store()
        try:
            await store.record_dispatch_event(
                task_id="task-empty",
                skill_ids=[],
                method="no_llm_prefilter",
            )
            with store._reader() as conn:
                rows = conn.execute(
                    "SELECT skill_ids FROM skill_dispatch_events WHERE task_id = ?",
                    ("task-empty",),
                ).fetchall()
            assert json.loads(rows[0]["skill_ids"]) == []
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_method_preserved(self):
        """Method string is stored verbatim."""
        store = make_temp_store()
        try:
            await store.record_dispatch_event(
                task_id="task-m",
                skill_ids=["s1"],
                method="no_llm_prefilter",
            )
            with store._reader() as conn:
                row = conn.execute(
                    "SELECT method FROM skill_dispatch_events",
                ).fetchone()
            assert row["method"] == "no_llm_prefilter"
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_clear_removes_dispatch_events(self):
        """store.clear() removes all dispatch event rows."""
        store = make_temp_store()
        try:
            await store.record_dispatch_event("t1", ["s1"], "llm")
            store.clear()
            with store._reader() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM skill_dispatch_events"
                ).fetchone()[0]
            assert count == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# tool_layer — _select_and_inject_skills dispatch event integration
# ---------------------------------------------------------------------------

class TestToolLayerDispatchEvent:
    def _make_tool_layer(self, with_store: bool = True):
        """Build a minimal mock OpenSpace instance for testing dispatch event."""
        from openspace.tool_layer import OpenSpace
        tl = object.__new__(OpenSpace)
        tl._initialized = True
        tl._running = False

        # Minimal mocks for _select_and_inject_skills
        mock_skill = MagicMock()
        mock_skill.skill_id = "skill-x"
        mock_skill.content = "skill content"

        registry = MagicMock()
        registry.select_skills_with_llm = AsyncMock(
            return_value=([mock_skill], {"method": "llm", "task": "do stuff", "selected": ["skill-x"]})
        )
        registry.list_skills = MagicMock(return_value=[mock_skill])
        registry.ts_blend_reorder = MagicMock(return_value=[mock_skill])
        registry.build_context_injection = MagicMock(return_value="injected context")

        tl._skill_registry = registry
        tl._grounding_agent = MagicMock()
        tl._grounding_config = MagicMock()
        tl._grounding_config.skills.max_select = 2
        tl._recording_manager = None

        if with_store:
            store = MagicMock()
            store.get_summary = MagicMock(return_value=[])
            store.get_bandit_stats = MagicMock(return_value={})
            store.record_dispatch_event = AsyncMock()
            tl._skill_store = store
        else:
            tl._skill_store = None

        tl.config = MagicMock()
        tl.config.skill_registry_model = None
        tl.config.tool_retrieval_model = None
        tl.config.llm_model = "test-model"
        tl._llm_client = MagicMock()
        tl._llm_client.model = "test-model"
        tl._get_skill_selection_llm = MagicMock(return_value=None)  # no-LLM path

        return tl

    @pytest.mark.asyncio
    async def test_dispatch_event_written_when_store_present(self):
        """record_dispatch_event is called when store is set and skills are selected."""
        tl = self._make_tool_layer(with_store=True)
        await tl._select_and_inject_skills("do stuff", task_id="task-123")
        tl._skill_store.record_dispatch_event.assert_called_once()
        call_args, call_kwargs = tl._skill_store.record_dispatch_event.call_args
        # task_id may be positional or keyword
        task_id_val = call_kwargs.get("task_id") or (call_args[0] if call_args else None)
        assert task_id_val == "task-123"

    @pytest.mark.asyncio
    async def test_dispatch_event_not_written_when_store_none(self):
        """No dispatch event when _skill_store is None."""
        tl = self._make_tool_layer(with_store=False)
        # Should not raise — just silently skips
        result = await tl._select_and_inject_skills("do stuff", task_id="task-no-store")
        # No store to assert on — test that no exception is raised
        assert result is True or result is False  # method completed

    @pytest.mark.asyncio
    async def test_dispatch_event_failure_is_nonfatal(self):
        """A store write failure doesn't propagate — method still returns True."""
        tl = self._make_tool_layer(with_store=True)
        tl._skill_store.record_dispatch_event = AsyncMock(
            side_effect=RuntimeError("DB write failed")
        )
        result = await tl._select_and_inject_skills("do stuff", task_id="task-err")
        assert result is True  # non-fatal — method returns True despite store error
