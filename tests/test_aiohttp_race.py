"""Tests for Wave 3a / Wave 4a — BaseConnector asyncio.Lock (race condition fixes).

Covers (Wave 3a — BaseConnector):
  - concurrent connect() calls only create one session (lock correctness)
  - second connect() is idempotent after first completes (no double-start)
  - concurrent disconnect() calls only stop manager once
  - high-concurrency (10 callers) still produces a single start()
  - _connect_lock attribute exists and is an asyncio.Lock

Covers (Wave 4a — AioHttpConnector ping-inside-lock):
  - ping failure while a concurrent caller is waiting leaves both callers
    with _connected=False (no half-ready session observable)
  - successful ping → both concurrent callers see _connected=True (idempotent)
  - ping failure triggers _cleanup_on_connect_failure (session closed), not disconnect()
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openspace.grounding.core.transport.connectors.base import BaseConnector
from openspace.grounding.core.transport.connectors.aiohttp_connector import AioHttpConnector
from openspace.grounding.core.transport.task_managers import BaseConnectionManager


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeConnectionManager(BaseConnectionManager):
    """Counts start()/stop() calls; yields to the event loop to expose races."""

    def __init__(self):
        super().__init__()
        self.start_count = 0
        self.stop_count = 0
        self._fake_session = object()

    async def _establish_connection(self):
        self.start_count += 1
        await asyncio.sleep(0)   # yield so concurrent callers can progress
        return self._fake_session

    async def _close_connection(self) -> None:
        self.stop_count += 1


class ConcreteConnector(BaseConnector):
    """Minimal concrete implementation for testing."""

    async def invoke(self, name: str, params: dict[str, Any]) -> Any:
        pass

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        pass


def make_connector() -> tuple[ConcreteConnector, FakeConnectionManager]:
    mgr = FakeConnectionManager()
    conn = ConcreteConnector(mgr)
    return conn, mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseConnectorLock:

    @pytest.mark.asyncio
    async def test_lock_attribute_exists(self):
        conn, _ = make_connector()
        assert hasattr(conn, "_connect_lock"), "_connect_lock must be set in __init__"
        assert isinstance(conn._connect_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_concurrent_connect_calls_start_only_once(self):
        """Two concurrent connect() calls must produce exactly one start()."""
        conn, mgr = make_connector()

        await asyncio.gather(conn.connect(), conn.connect())

        assert mgr.start_count == 1, (
            f"Expected 1 start() call; got {mgr.start_count}. "
            "Race condition: both callers passed the _connected check before either set it."
        )
        assert conn.is_connected

    @pytest.mark.asyncio
    async def test_sequential_connect_is_idempotent(self):
        """connect() after already connected must be a no-op."""
        conn, mgr = make_connector()

        await conn.connect()
        await conn.connect()

        assert mgr.start_count == 1

    @pytest.mark.asyncio
    async def test_10_concurrent_connects_start_once(self):
        """High concurrency: 10 simultaneous connect() calls → exactly one start()."""
        conn, mgr = make_connector()

        await asyncio.gather(*[conn.connect() for _ in range(10)])

        assert mgr.start_count == 1
        assert conn.is_connected

    @pytest.mark.asyncio
    async def test_concurrent_disconnect_stops_once(self):
        """Two concurrent disconnect() calls must call stop() exactly once."""
        conn, mgr = make_connector()
        await conn.connect()

        await asyncio.gather(conn.disconnect(), conn.disconnect())

        assert mgr.stop_count == 1
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_connect_then_disconnect_then_reconnect(self):
        """Full lifecycle: connect → disconnect → reconnect must work correctly."""
        conn, mgr = make_connector()

        await conn.connect()
        assert conn.is_connected
        assert mgr.start_count == 1

        await conn.disconnect()
        assert not conn.is_connected
        assert mgr.stop_count == 1

        await conn.connect()
        assert conn.is_connected
        assert mgr.start_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_is_noop(self):
        """disconnect() on an unconnected connector must not call stop()."""
        conn, mgr = make_connector()

        await conn.disconnect()

        assert mgr.stop_count == 0


# ---------------------------------------------------------------------------
# AioHttpConnector ping-inside-lock race tests (Wave 4a)
# ---------------------------------------------------------------------------

class TestAioHttpConnectorPingRace:
    """Verify that the _after_connect ping fires inside the _connect_lock so no
    concurrent caller can observe a connected=True state before the ping succeeds.
    """

    def _make_aiohttp_connector(self, ping_raises=None, ping_status=200):
        """Build an AioHttpConnector whose HTTP ping is intercepted."""
        conn = AioHttpConnector.__new__(AioHttpConnector)
        # Minimal __init__ via BaseConnector
        mgr = MagicMock()
        mgr.start = AsyncMock(return_value=MagicMock())  # fake aiohttp session
        mgr.stop = AsyncMock()
        BaseConnector.__init__(conn, mgr)
        conn.base_url = "http://fake"

        # Patch _after_connect at instance level so we control the ping
        if ping_raises:
            async def bad_ping():
                raise ping_raises
            conn._after_connect = bad_ping
        else:
            async def good_ping():
                pass
            conn._after_connect = good_ping

        return conn, mgr

    @pytest.mark.asyncio
    async def test_ping_failure_leaves_connected_false(self):
        """If the ping fails, _connected must remain False after connect() raises."""
        conn, mgr = self._make_aiohttp_connector(
            ping_raises=ConnectionError("ping failed")
        )

        with pytest.raises(ConnectionError):
            await conn.connect()

        assert not conn.is_connected, (
            "_connected must stay False when _after_connect raises"
        )
        mgr.stop.assert_awaited_once()  # _cleanup_on_connect_failure must close session

    @pytest.mark.asyncio
    async def test_concurrent_connect_ping_failure_both_callers_see_disconnected(self):
        """Concurrent callers: if the first caller's ping fails, the second caller
        must NOT observe _connected=True at any point during or after the attempt."""
        conn, mgr = self._make_aiohttp_connector(
            ping_raises=ConnectionError("ping failed")
        )

        results = []

        async def try_connect():
            try:
                await conn.connect()
                results.append("connected")
            except ConnectionError:
                results.append("failed")

        await asyncio.gather(try_connect(), try_connect())

        # Both callers must see failure; no caller should slip through with connected=True
        assert all(r == "failed" for r in results), (
            f"Expected all callers to fail when ping fails; got {results}"
        )
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_concurrent_connect_ping_success_all_callers_see_connected(self):
        """When the ping succeeds, all concurrent callers must observe is_connected=True."""
        conn, mgr = self._make_aiohttp_connector()  # good ping

        await asyncio.gather(*[conn.connect() for _ in range(5)])

        assert conn.is_connected
        assert mgr.start.await_count == 1, (
            "Session must be started exactly once despite concurrent callers"
        )
