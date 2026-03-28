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
    async def test_same_label_does_not_overlap_runs_sequentially(self):
        """Second call with same in-flight label must NOT run concurrently.
        It is queued as a pending rerun and executes only after the first completes."""
        evolver = make_evolver()
        concurrent_count = 0
        max_concurrent = 0
        barrier = asyncio.Event()

        async def work():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await barrier.wait()
            concurrent_count -= 1

        t1 = evolver.schedule_background(work(), label="tool_degradation")
        evolver.schedule_background(work(), label="tool_degradation")  # queued as pending

        barrier.set()
        if t1:
            await t1
        await asyncio.sleep(0)  # let pending rerun start
        await asyncio.sleep(0)  # let pending rerun complete

        assert max_concurrent == 1, (
            "The two coroutines must never execute concurrently; "
            f"got max_concurrent={max_concurrent}"
        )

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
        """schedule_background without a running event loop must return None.

        The coroutine must be closed by production code (not left open) to prevent
        'coroutine was never awaited' ResourceWarning.  We verify by checking that
        the coro is in a closed state after the call.
        """
        evolver = make_evolver()

        async def work():
            pass  # pragma: no cover

        coro = work()
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            result = evolver.schedule_background(coro, label="test")

        assert result is None
        # A closed coroutine has cr_frame=None (reliable across Python versions).
        assert coro.cr_frame is None, (
            "Production code must close the coroutine to prevent ResourceWarning"
        )

    @pytest.mark.asyncio
    async def test_pending_rerun_queued_when_label_in_flight(self):
        """A same-label coro submitted while first is running must be stored as pending rerun."""
        evolver = make_evolver()
        barrier = asyncio.Event()
        ran: list[int] = []

        async def slow():
            await barrier.wait()
            ran.append(1)

        async def rerun():
            ran.append(2)

        t1 = evolver.schedule_background(slow(), label="tool_degradation")
        t2 = evolver.schedule_background(rerun(), label="tool_degradation")

        assert t1 is not None
        assert t2 is None, "Duplicate in-flight label must still return None"
        assert "tool_degradation" in evolver._pending_reruns, (
            "Latest coro must be stored as pending rerun, not discarded"
        )

        barrier.set()
        await t1
        await asyncio.sleep(0)  # let done-callbacks fire
        await asyncio.sleep(0)  # let pending task start + complete

        assert ran == [1, 2], "Pending rerun must execute after predecessor completes"
        assert "tool_degradation" not in evolver._pending_reruns

    @pytest.mark.asyncio
    async def test_pending_rerun_last_write_wins(self):
        """Multiple same-label calls while in-flight: only the last coro is kept."""
        evolver = make_evolver()
        barrier = asyncio.Event()
        ran: list[int] = []

        async def slow():
            await barrier.wait()

        async def second():
            ran.append(2)

        async def third():
            ran.append(3)

        t1 = evolver.schedule_background(slow(), label="tool_degradation")
        evolver.schedule_background(second(), label="tool_degradation")  # queued
        evolver.schedule_background(third(), label="tool_degradation")  # replaces second

        assert "tool_degradation" in evolver._pending_reruns

        barrier.set()
        await t1
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Only third() ran — second() was replaced and its coro closed
        assert ran == [3], "Last submitted coro must win; stale pending must be closed"

    @pytest.mark.asyncio
    async def test_superseded_pending_coro_is_closed(self):
        """When a third same-label call replaces the pending rerun, the OLD pending
        coro must be closed (last-write-wins cleanup, no leak)."""
        evolver = make_evolver()
        barrier = asyncio.Event()
        closed_second = False

        class TrackClose:
            """Wraps a coroutine and records whether close() was called."""

            def __init__(self, inner):
                self._inner = inner

            def __await__(self):
                return self._inner.__await__()

            def close(self):
                nonlocal closed_second
                closed_second = True
                self._inner.close()

            def send(self, v):
                return self._inner.send(v)

            def throw(self, *a):
                return self._inner.throw(*a)

        async def slow():
            await barrier.wait()

        async def second_coro():
            pass  # pragma: no cover — gets replaced before it can run

        async def third_coro():
            pass

        t1 = evolver.schedule_background(slow(), label="dup_test")
        # Second call: stored as pending rerun (NOT immediately closed)
        evolver.schedule_background(TrackClose(second_coro()), label="dup_test")
        assert not closed_second, "Pending rerun must not be closed on first duplicate"

        # Third call: replaces second; second's coro must be closed now
        evolver.schedule_background(third_coro(), label="dup_test")
        assert closed_second, "Superseded pending rerun must be closed when replaced"

        barrier.set()
        if t1:
            await t1
        await asyncio.sleep(0)  # let third_coro run as pending rerun
        await asyncio.sleep(0)
