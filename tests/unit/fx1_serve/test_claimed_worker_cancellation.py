"""SYNTHETIC cancellation barriers using the real proposed helper and registry."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import anyio
import pytest

from fx1.serve.api import ApiError, _run_claimed
from fx1.serve.journal import _ClaimLocks


async def _event(event: threading.Event) -> None:
    with anyio.fail_after(2):
        while not event.is_set():
            await anyio.sleep(0.001)


async def _users(claims: _ClaimLocks, count: int) -> None:
    with anyio.fail_after(2):
        while claims._locks["same"].users != count:
            await anyio.sleep(0.001)


@pytest.mark.parametrize("cancellations", [1, 5])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_cancelled_leader_keeps_reservation_until_worker_finishes(
    cancellations: int, worker_fails: bool
) -> None:
    async def exercise() -> None:
        claims = _ClaimLocks(1)
        entered, release, exited, retry_entered = (threading.Event() for _ in range(4))

        def work() -> int:
            entered.set()
            try:
                assert release.wait(3), "test did not release worker"
                if worker_fails:
                    raise ApiError(422, "synthetic cancelled worker failure")
                return 17
            finally:
                exited.set()

        def retry() -> int:
            assert exited.is_set(), "retry overlapped original worker"
            retry_entered.set()
            return 29

        leader = asyncio.create_task(_run_claimed(claims.ahold("same"), work))
        follower: asyncio.Task[int] | None = None
        try:
            await _event(entered)
            original = claims._locks["same"]
            leader.cancel()
            follower = asyncio.create_task(_run_claimed(claims.ahold("same"), retry))
            await _users(claims, 2)
            for _ in range(cancellations - 1):
                await anyio.sleep(0.005)
                leader.cancel()
            # Exercise registry pressure while both holder and waiter are pinned.
            async with claims.ahold("other"):
                assert claims._locks["same"] is original
            await anyio.sleep(0.02)
            assert not leader.done()
            assert not retry_entered.is_set()
            assert original.users == 2
            assert original.lock.locked()
        finally:
            release.set()
            answers = await asyncio.gather(
                *([leader, follower] if follower is not None else [leader]),
                return_exceptions=True,
            )
        assert isinstance(answers[0], asyncio.CancelledError)
        assert answers[1] == 29
        assert exited.is_set()
        assert retry_entered.is_set()
        assert len(claims._locks) <= 1
        assert all(claim.users == 0 and not claim.lock.locked() for claim in claims._locks.values())

    asyncio.run(exercise())


def test_cancelled_waiter_releases_only_its_reservation() -> None:
    async def exercise() -> None:
        claims = _ClaimLocks(1)
        called: list[str] = []
        async with claims.ahold("same"):
            original = claims._locks["same"]
            waiter = asyncio.create_task(
                _run_claimed(claims.ahold("same"), lambda: called.append("cancelled waiter"))
            )
            await _users(claims, 2)
            waiter.cancel()
            answers = await asyncio.gather(waiter, return_exceptions=True)
            assert isinstance(answers[0], asyncio.CancelledError)
            assert original.users == 1
            assert original.lock.locked()
            assert not called
        assert original.users == 0
        assert not original.lock.locked()
        assert await _run_claimed(claims.ahold("same"), lambda: 31) == 31
        assert original.users == 0

    asyncio.run(exercise())


def test_anyio_scope_cancellation_drains_worker_before_releasing_claim() -> None:
    async def exercise() -> None:
        claims = _ClaimLocks(1)
        entered, release, exited = (threading.Event() for _ in range(3))
        leader_exited = anyio.Event()
        scope = anyio.CancelScope()

        def work() -> None:
            entered.set()
            try:
                assert release.wait(3), "test did not release worker"
            finally:
                exited.set()

        async def leader() -> None:
            with scope:
                try:
                    await _run_claimed(claims.ahold("same"), work)
                finally:
                    assert exited.is_set()
                    leader_exited.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(leader)
            await _event(entered)
            scope.cancel()
            try:
                await anyio.sleep(0.02)
                assert not leader_exited.is_set()
                assert claims._locks["same"].users == 1
                assert claims._locks["same"].lock.locked()
            finally:
                release.set()
        assert leader_exited.is_set()
        assert claims._locks["same"].users == 0
        assert not claims._locks["same"].lock.locked()

    anyio.run(exercise)


def test_worker_result_and_http_error_keep_original_types() -> None:
    async def exercise() -> None:
        claims = _ClaimLocks(1)
        error = ApiError(422, "synthetic worker failure")

        def fail() -> Any:
            raise error

        assert await _run_claimed(claims.ahold("same"), lambda: 23) == 23
        assert await _run_claimed(claims.ahold("same"), lambda: None) is None
        with pytest.raises(ApiError) as caught:
            await _run_claimed(claims.ahold("same"), fail)
        assert caught.value is error
        assert claims._locks["same"].users == 0
        assert not claims._locks["same"].lock.locked()
        assert await _run_claimed(claims.ahold("same"), lambda: 37) == 37

    anyio.run(exercise)
