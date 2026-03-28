"""Tests for Wave 3b — SkillEvolver schedule_background deduplication.

Covers:
  - duplicate label → second coroutine discarded, first task returned
  - different labels → both tasks run (no false dedup)
  - after first task completes, same label is accepted again
  - discarded coroutine is properly closed (no ResourceWarning)
  - no running event loop → returns None
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.skill_engine.evolver import SkillEvolver


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_evolver() -> SkillEvolver:
    """Minimal SkillEvolver — store/registry/llm not needed for these tests."""
    store = MagicMock()
    registry = MagicMock()
    llm_client = AsyncMock()
    return SkillEvolver(store=store, registry=registry, llm_client=llm_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScheduleBackgroundDedup:

    @pytest.mark.asyncio
    async def test_same_label_returns_none_for_second_call(self):
        """Second schedule_background() with same label while first is running must return None."""
        evolver = make_evolver()
        barrier = asyncio.Event()

        async def slow_work():
            await barrier.wait()   # block until we release

        t1 = evolver.schedule_background(slow_work(), label="tool_degradation")
        t2 = evolver.schedule_background(slow_work(), label="tool_degradation")

        assert t1 is not None
        assert t2 is None, "Duplicate in-flight label must return None"

        barrier.set()   # let t1 finish
        await t1

    @pytest.mark.asyncio
    async def test_same_label_only_runs_first_coroutine(self):
        """Only the first coroutine executes; the second is discarded."""
        evolver = make_evolver()
        call_count = 0
        barrier = asyncio.Event()

        async def work():
            nonlocal call_count
            await barrier.wait()
            call_count += 1

        t1 = evolver.schedule_background(work(), label="tool_degradation")
        evolver.schedule_background(work(), label="tool_degradation")  # discarded

        barrier.set()
        if t1:
            await t1
        await asyncio.sleep(0)

        assert call_count == 1, "Discarded coroutine must not execute"

    @pytest.mark.asyncio
    async def test_different_labels_both_scheduled(self):
        """Two calls with different labels must both be scheduled and run."""
        evolver = make_evolver()
        ran: list[str] = []

        async def work(name: str):
            await asyncio.sleep(0)
            ran.append(name)

        t1 = evolver.schedule_background(work("alpha"), label="alpha")
        t2 = evolver.schedule_background(work("beta"), label="beta")

        assert t1 is not None
        assert t2 is not None

        await asyncio.gather(t1, t2)
        assert sorted(ran) == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_same_label_accepted_after_first_completes(self):
        """Once the first task finishes, a same-label task must be accepted."""
        evolver = make_evolver()
        call_count = 0

        async def work():
            nonlocal call_count
            call_count += 1

        t1 = evolver.schedule_background(work(), label="tool_degradation")
        assert t1 is not None
        await t1   # let first finish; it will be removed from _background_tasks via done callback

        t2 = evolver.schedule_background(work(), label="tool_degradation")
        assert t2 is not None, "After first completes, same label must be accepted again"
        await t2

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_running_loop_returns_none(self):
        """schedule_background without a running event loop must return None."""
        evolver = make_evolver()

        async def work():
            pass  # pragma: no cover

        coro = work()
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            result = evolver.schedule_background(coro, label="test")
        coro.close()  # prevent ResourceWarning

        assert result is None

    @pytest.mark.asyncio
    async def test_discarded_coroutine_is_closed(self):
        """Discarded duplicate coroutine must be closed (no ResourceWarning)."""
        evolver = make_evolver()
        barrier = asyncio.Event()
        closed = False

        class TrackClose:
            """Wraps a coroutine and records whether close() was called."""

            def __init__(self, inner):
                self._inner = inner

            def __await__(self):
                return self._inner.__await__()

            def close(self):
                nonlocal closed
                closed = True
                self._inner.close()

            def send(self, v):
                return self._inner.send(v)

            def throw(self, *a):
                return self._inner.throw(*a)

        async def slow():
            await barrier.wait()

        async def dummy():
            pass

        t1 = evolver.schedule_background(slow(), label="dup_test")
        evolver.schedule_background(TrackClose(dummy()), label="dup_test")

        assert closed, "Discarded coroutine's close() must be called"

        barrier.set()
        if t1:
            await t1
