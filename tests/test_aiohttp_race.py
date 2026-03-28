"""Tests for Wave 3a — BaseConnector asyncio.Lock (race condition fix).

Covers:
  - concurrent connect() calls only create one session (lock correctness)
  - second connect() is idempotent after first completes (no double-start)
  - concurrent disconnect() calls only stop manager once
  - high-concurrency (10 callers) still produces a single start()
  - _connect_lock attribute exists and is an asyncio.Lock
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from openspace.grounding.core.transport.connectors.base import BaseConnector
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
