"""Regression checks for journal and idempotency synchronization."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from fx1.serve.journal import _ClaimLocks


def test_claim_cannot_evict_itself_while_capacity_is_busy() -> None:
    """Distinct active keys may exceed the cache bound, never split one mutex."""
    claims = _ClaimLocks(1)
    attempted = threading.Event()
    entered = threading.Event()

    def contender() -> None:
        attempted.set()
        with claims.hold("second"):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool, claims.hold("first"):
        with claims.hold("second"):
            future = pool.submit(contender)
            assert attempted.wait(2)
            overlapped = entered.wait(0.2)
        future.result(timeout=2)
    assert entered.is_set()
    assert not overlapped, "same-key claims entered concurrently after cache saturation"


def test_claim_is_reusable_after_handler_failure() -> None:
    claims = _ClaimLocks(1)
    with pytest.raises(RuntimeError, match="handler failed"), claims.hold("key"):
        raise RuntimeError("handler failed")
    with claims.hold("other"), claims.hold("key"):
        pass


def test_missing_idempotency_key_never_serializes() -> None:
    claims = _ClaimLocks(1)
    entered = threading.Event()

    def contender() -> None:
        with claims.hold(None):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with claims.hold(None):
            future = pool.submit(contender)
            assert entered.wait(2)
        future.result(timeout=2)


def test_async_claim_waits_without_consuming_request_workers() -> None:
    import anyio

    claims = _ClaimLocks(1)

    async def exercise() -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        entered = asyncio.Event()

        async def waiter() -> None:
            async with claims.ahold("same"):
                entered.set()

        with claims.hold("same"):
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.02)
            assert not entered.is_set()
            assert (
                await asyncio.wait_for(anyio.to_thread.run_sync(lambda: "progress"), 2)
                == "progress"
            )
        await asyncio.wait_for(task, 2)
        assert entered.is_set()

    asyncio.run(exercise())


def test_cancelled_async_waiter_releases_its_reservation() -> None:
    claims = _ClaimLocks(1)

    async def exercise() -> None:
        entered = asyncio.Event()

        async def waiter() -> None:
            async with claims.ahold("same"):
                entered.set()

        with claims.hold("same"):
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not entered.is_set()
        with claims.hold("other"):
            async with claims.ahold("same"):
                pass

    asyncio.run(exercise())
