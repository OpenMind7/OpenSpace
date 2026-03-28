"""Tests for W6 Step 12 (bandit_snapshot) and Step 13 (bandit decay)."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from openspace.skill_engine.store import SkillStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_store() -> SkillStore:
    tmp = tempfile.mkdtemp()
    return SkillStore(db_path=Path(tmp) / "test_bandit.db")


def seed_bandit(store: SkillStore, skill_id: str, alpha: float, beta: float,
                dispatches: int = 5) -> None:
    with store._mu:
        store._conn.execute(
            "INSERT OR REPLACE INTO skill_bandit "
            "(skill_id, alpha, beta, prior_confidence, total_dispatches, last_updated) "
            "VALUES (?, ?, ?, 0.5, ?, ?)",
            (skill_id, alpha, beta, dispatches, datetime.now().isoformat()),
        )
        store._conn.commit()


def read_dispatch_row(store: SkillStore, task_id: str) -> Dict[str, Any]:
    with store._reader() as conn:
        row = conn.execute(
            "SELECT * FROM skill_dispatch_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Step 12 — DDL: bandit_snapshot column
# ---------------------------------------------------------------------------

class TestBanditSnapshotDDL:
    def test_column_exists_in_schema(self):
        store = make_temp_store()
        try:
            with store._reader() as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(skill_dispatch_events)").fetchall()]
            assert "bandit_snapshot" in cols
        finally:
            store.close()

    def test_column_default_is_empty_json(self):
        store = make_temp_store()
        try:
            with store._mu:
                store._conn.execute(
                    "INSERT INTO skill_dispatch_events "
                    "(task_id, skill_ids, method, dispatched_at) "
                    "VALUES ('t0', '[]', 'test', '2026-01-01')"
                )
                store._conn.commit()
            row = read_dispatch_row(store, "t0")
            assert row["bandit_snapshot"] == "{}"
        finally:
            store.close()


class TestMigration:
    def test_reinit_does_not_raise(self):
        store = make_temp_store()
        try:
            store._init_db()  # second call — ALTER TABLE silently ignored
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Step 12 — record_dispatch_event with bandit_snapshot
# ---------------------------------------------------------------------------

class TestRecordDispatchEventWithSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_stored_when_provided(self):
        store = make_temp_store()
        try:
            snapshot = {"skill-a": {"alpha": 2.5, "beta": 1.3}, "skill-b": {"alpha": 1.0, "beta": 4.0}}
            await store.record_dispatch_event(
                "task-1", ["skill-a", "skill-b"], "llm", bandit_snapshot=snapshot
            )
            row = read_dispatch_row(store, "task-1")
            stored = json.loads(row["bandit_snapshot"])
            assert stored["skill-a"]["alpha"] == pytest.approx(2.5)
            assert stored["skill-b"]["beta"] == pytest.approx(4.0)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_snapshot_none_stores_empty_dict(self):
        store = make_temp_store()
        try:
            await store.record_dispatch_event("task-2", ["s1"], "llm", bandit_snapshot=None)
            row = read_dispatch_row(store, "task-2")
            assert json.loads(row["bandit_snapshot"]) == {}
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_snapshot_omitted_stores_empty_dict(self):
        store = make_temp_store()
        try:
            await store.record_dispatch_event("task-3", ["s1"], "keyword_only")
            row = read_dispatch_row(store, "task-3")
            assert json.loads(row["bandit_snapshot"]) == {}
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_skill_ids_still_correct(self):
        store = make_temp_store()
        try:
            await store.record_dispatch_event(
                "task-4", ["s1", "s2"], "llm",
                bandit_snapshot={"s1": {"alpha": 1.5, "beta": 2.0}}
            )
            row = read_dispatch_row(store, "task-4")
            assert json.loads(row["skill_ids"]) == ["s1", "s2"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Step 13 — decay_bandit_posteriors
# ---------------------------------------------------------------------------

class TestDecayBanditPosteriors:
    @pytest.mark.asyncio
    async def test_decay_shrinks_toward_prior(self):
        store = make_temp_store()
        try:
            seed_bandit(store, "sk-x", alpha=5.0, beta=3.0)
            updated = await store.decay_bandit_posteriors(decay_factor=0.9)
            assert updated == 1
            stats = store.get_bandit_stats(["sk-x"])
            # alpha: 1 + (5-1)*0.9 = 4.6 ; beta: 1 + (3-1)*0.9 = 2.8
            assert stats["sk-x"].alpha == pytest.approx(4.6, rel=1e-5)
            assert stats["sk-x"].beta == pytest.approx(2.8, rel=1e-5)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_preserves_beta_1_1_fixed_point(self):
        """Beta(1,1) is the fixed point — decay must leave it unchanged."""
        store = make_temp_store()
        try:
            seed_bandit(store, "sk-cold", alpha=1.0, beta=1.0)
            await store.decay_bandit_posteriors(decay_factor=0.99)
            stats = store.get_bandit_stats(["sk-cold"])
            assert stats["sk-cold"].alpha == pytest.approx(1.0)
            assert stats["sk-cold"].beta == pytest.approx(1.0)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_skips_zero_dispatch_skills(self):
        """Skills never dispatched (total_dispatches=0) are not decayed."""
        store = make_temp_store()
        try:
            seed_bandit(store, "sk-new", alpha=3.0, beta=2.0, dispatches=0)
            updated = await store.decay_bandit_posteriors()
            assert updated == 0
            stats = store.get_bandit_stats(["sk-new"])
            assert stats["sk-new"].alpha == pytest.approx(3.0)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_multiple_skills(self):
        store = make_temp_store()
        try:
            seed_bandit(store, "sk-a", alpha=3.0, beta=2.0)
            seed_bandit(store, "sk-b", alpha=5.0, beta=5.0)
            updated = await store.decay_bandit_posteriors(decay_factor=0.5)
            assert updated == 2
            stats = store.get_bandit_stats(["sk-a", "sk-b"])
            # sk-a alpha: 1+(3-1)*0.5 = 2.0 ; beta: 1+(2-1)*0.5 = 1.5
            assert stats["sk-a"].alpha == pytest.approx(2.0)
            assert stats["sk-a"].beta == pytest.approx(1.5)
            # sk-b alpha: 1+(5-1)*0.5 = 3.0 ; beta: 1+(5-1)*0.5 = 3.0
            assert stats["sk-b"].alpha == pytest.approx(3.0)
            assert stats["sk-b"].beta == pytest.approx(3.0)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_returns_zero_when_empty(self):
        store = make_temp_store()
        try:
            updated = await store.decay_bandit_posteriors()
            assert updated == 0
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_default_decay_factor_is_0_99(self):
        store = make_temp_store()
        try:
            seed_bandit(store, "sk-d", alpha=101.0, beta=1.0)
            await store.decay_bandit_posteriors()
            stats = store.get_bandit_stats(["sk-d"])
            # alpha: 1 + (101-1)*0.99 = 100.0
            assert stats["sk-d"].alpha == pytest.approx(100.0, rel=1e-4)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Step 13 — evolver._analysis_count + decay scheduling
# ---------------------------------------------------------------------------

def make_evolver(store: SkillStore):
    from openspace.skill_engine.evolver import SkillEvolver
    registry = MagicMock()
    registry.ranker = MagicMock()
    registry.ranker.fine_tune_from_outcomes = AsyncMock(return_value=False)
    llm = MagicMock()
    return SkillEvolver(store=store, registry=registry, llm_client=llm)


def make_analysis(task_id: str = "t1"):
    a = MagicMock()
    a.task_completed = True
    a.skill_judgments = []
    a.causal_attributions = []
    a.evolution_suggestions = []
    a.task_id = task_id
    return a


class TestEvolverDecayTrigger:
    def test_analysis_count_initialized_to_zero(self):
        store = make_temp_store()
        try:
            evolver = make_evolver(store)
            assert evolver._analysis_count == 0
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_analysis_count_increments_each_call(self):
        store = make_temp_store()
        try:
            evolver = make_evolver(store)
            # patch schedule_background to avoid background task leak
            evolver.schedule_background = lambda coro, *, label=None: (
                coro.close() if hasattr(coro, "close") else None
            )
            await evolver.process_analysis(make_analysis("t1"))
            assert evolver._analysis_count == 1
            await evolver.process_analysis(make_analysis("t2"))
            assert evolver._analysis_count == 2
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_scheduled_at_multiple_of_100(self):
        store = make_temp_store()
        try:
            evolver = make_evolver(store)
            evolver._analysis_count = 99  # one away from trigger

            scheduled_labels = []

            def mock_schedule(coro, *, label=None):
                scheduled_labels.append(label)
                if hasattr(coro, "close"):
                    coro.close()
                return None

            evolver.schedule_background = mock_schedule
            await evolver.process_analysis(make_analysis())
            assert "bandit_decay" in scheduled_labels
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_decay_not_scheduled_before_100(self):
        store = make_temp_store()
        try:
            evolver = make_evolver(store)
            evolver._analysis_count = 98  # two away from trigger

            scheduled_labels = []

            def mock_schedule(coro, *, label=None):
                scheduled_labels.append(label)
                if hasattr(coro, "close"):
                    coro.close()
                return None

            evolver.schedule_background = mock_schedule
            await evolver.process_analysis(make_analysis())
            assert "bandit_decay" not in scheduled_labels
        finally:
            store.close()
