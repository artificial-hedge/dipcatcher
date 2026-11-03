"""Claim and journal locks fail closed when a holder never releases."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fx1.serve.journal import JobJournal, _ClaimLocks


def test_nested_claim_times_out_and_stays_reusable() -> None:
    claims = _ClaimLocks(1, wait_s=0.05)
    with claims.hold("same"), pytest.raises(TimeoutError, match="timed out"), claims.hold("same"):
        pass
    with claims.hold("same"):
        pass
    assert claims._locks["same"].users == 0
    assert not claims._locks["same"].lock.locked()


def test_async_claim_times_out_without_keeping_the_reservation() -> None:
    claims = _ClaimLocks(1, wait_s=0.05)

    async def exercise() -> None:
        with claims.hold("same"):

            async def wait() -> None:
                async with claims.ahold("same"):
                    pass

            with pytest.raises(TimeoutError, match="timed out"):
                await wait()
        async with claims.ahold("same"):
            pass

    asyncio.run(exercise())
    assert claims._locks["same"].users == 0
    assert not claims._locks["same"].lock.locked()


def test_journal_append_times_out_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "jobs.jsonl"
    journal = JobJournal(path, wait_s=0.05)
    assert journal._lock.acquire(timeout=0.1)
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            journal.append({"id": 1})
    finally:
        journal._lock.release()
    assert not path.exists()
    journal.append({"id": 1})
    assert journal.replay().payloads == [{"id": 1}]


def test_lock_wait_rejects_non_positive_deadlines() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _ClaimLocks(1, wait_s=0)
    with pytest.raises(ValueError, match="positive finite"):
        JobJournal("jobs.jsonl", wait_s=float("inf"))
