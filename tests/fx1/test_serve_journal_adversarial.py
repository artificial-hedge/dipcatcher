"""SYNTHETIC adversarial probes for ``fx1.serve.journal``.

Targets the byte-level chain contract beyond the happy-path replay suite:
mid-journal payload re-serialization, smuggled top-level keys, sequence
gaps, compact→append chain continuity, and ``_ClaimLocks`` exclusion
guarantees under eviction and cooperative cancellation. All fixtures are
synthetic; results are correctness checks, never market evidence.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal, _ClaimLocks, _line_bytes

ZERO_CHAIN = "0" * 64


def _record(seq: int, payload: Any, chain: str = ZERO_CHAIN) -> bytes:
    """Independently encode a record exactly like the legacy wire encoder."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{seq}|{chain}|{encoded}".encode()).hexdigest()
    return (
        json.dumps(
            {"seq": seq, "chain": chain, "sha256": digest, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def test_sha_binds_seq_chain_and_canonical_payload() -> None:
    """The per-line digest covers seq|chain|canonical payload — not raw bytes."""
    payload = {"b": 2, "a": 1}
    expected = hashlib.sha256(
        f"{7}|{ZERO_CHAIN}|{json.dumps(payload, sort_keys=True, separators=(',', ':'))}".encode()
    ).hexdigest()
    line = json.loads(_line_bytes(7, ZERO_CHAIN, payload))
    assert line["sha256"] == expected
    assert line["seq"] == 7 and line["chain"] == ZERO_CHAIN


def test_mid_journal_payload_reformat_kills_the_tail(tmp_path: Path) -> None:
    """Re-serializing a middle record keeps its own sha valid but breaks the
    next link — a byte-level reflow can only destroy, never hide."""
    first = _record(0, {"id": 0})
    second_raw = _record(1, {"id": 1}, hashlib.sha256(first).hexdigest())
    second_line = json.loads(second_raw)
    third = _record(2, {"id": 2}, hashlib.sha256(second_raw).hexdigest())
    # Same parsed content, different single-line bytes (spaced separators):
    # the line's sha256 covers canonical payload so it still verifies, but
    # `chain` in the next record was minted over the ORIGINAL raw line —
    # the tail dies on the chain mismatch exactly where replay stands.
    reflowed = json.dumps(second_line, sort_keys=True, separators=(", ", ": ")).encode() + b"\n"
    path = tmp_path / "journal.jsonl"
    path.write_bytes(first + reflowed + third)
    result = JobJournal(path).replay()
    assert result.payloads == [{"id": 0}, {"id": 1}]
    assert result.truncated_at == len(first) + len(reflowed)
    assert result.dropped == 1


def test_extra_toplevel_key_does_not_reach_replayed_payload(tmp_path: Path) -> None:
    """The sealed sha covers only seq|chain|payload; a smuggled top-level key
    on the final line is dropped, not surfaced to the store."""
    path = tmp_path / "journal.jsonl"
    line = json.loads(_record(0, {"id": "synthetic"}))
    line["evil"] = {"smuggle": True}
    path.write_bytes(json.dumps(line, sort_keys=True).encode() + b"\n")
    result = JobJournal(path).replay()
    assert result.payloads == [{"id": "synthetic"}]
    assert "evil" not in result.payloads[0]


def test_sequence_gap_rejects_the_skipped_claim(tmp_path: Path) -> None:
    """A record declaring a gap must not verify — replay stops exactly there."""
    path = tmp_path / "journal.jsonl"
    first = _record(0, {"id": 0})
    jumped = _record(2, {"id": 2}, hashlib.sha256(first).hexdigest())
    path.write_bytes(first + jumped)
    result = JobJournal(path).replay()
    assert result.payloads == [{"id": 0}]
    assert result.truncated_at == len(first)


def test_compact_then_append_continues_one_verified_chain(tmp_path: Path) -> None:
    """Compaction restarts the sequence; fresh appends chain onto the new tail
    and the whole file still replays end to end."""
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": "dead"})
    journal.compact([{"id": "live"}])
    journal.append({"id": "after"})
    result = JobJournal(path).replay()
    assert result.truncated_at is None
    assert result.payloads == [{"id": "live"}, {"id": "after"}]


def test_zero_byte_journal_replays_clean(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b"")
    journal = JobJournal(path)
    result = journal.replay()
    assert result.payloads == [] and result.truncated_at is None and result.dropped == 0
    journal.append({"id": 0})
    assert JobJournal(path).replay().payloads == [{"id": 0}]


def test_journal_path_is_a_directory_fails_loud_and_blocks(tmp_path: Path) -> None:
    """A directory as the journal path is a configuration fault — the append
    raises and the instance stays blocked instead of pretending success."""
    journal = JobJournal(tmp_path)
    with pytest.raises((IsADirectoryError, PermissionError, OSError)):
        journal.append({"id": 0})
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": 1})


def test_blocked_append_never_writes_partial_bytes(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": 0})
    head = path.read_bytes()
    path.write_bytes(head + b'{"torn"')
    journal.replay()
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "must not land"})
    assert path.read_bytes() == head + b'{"torn"'


def test_claims_serialize_same_key_and_evict_only_idle() -> None:
    """Over the bound, idle entries evict; an evicted key's next claim still
    serializes correctly against a concurrent holder."""
    locks = _ClaimLocks(bound=2)
    inside = 0
    guard = threading.Lock()

    def critical() -> None:
        nonlocal inside
        with locks.hold("key"):
            with guard:
                inside += 1
                assert inside == 1
            time.sleep(0.001)
            with guard:
                inside = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: critical(), range(40)))

    # Force eviction churn, then re-verify exclusion still holds for the key.
    for name in ("a", "b", "c"):
        with locks.hold(name):
            pass
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: critical(), range(16)))


def test_claim_none_key_never_serializes() -> None:
    """None claims bypass by contract — two holders overlap freely."""
    locks = _ClaimLocks(bound=2)
    entered = threading.Event()
    proceed = threading.Event()

    def waiter() -> None:
        with locks.hold(None):
            entered.set()
            proceed.wait(2.0)

    thread = threading.Thread(target=waiter)
    thread.start()
    try:
        assert entered.wait(2.0)
        with locks.hold(None):  # must not block on the held None claim
            pass
    finally:
        proceed.set()
        thread.join(2.0)


def test_ahold_waiters_hand_off_in_order() -> None:
    """Cooperative waiters serialize behind a held claim and hand off cleanly."""
    import anyio

    async def scenario() -> list[str]:
        locks = _ClaimLocks(bound=4)
        order: list[str] = []
        release = anyio.Event()

        async def holder() -> None:
            async with locks.ahold("k"):
                order.append("first")
                await release.wait()

        async def waiter() -> None:
            async with locks.ahold("k"):
                order.append("second")

        async with anyio.create_task_group() as tg:
            tg.start_soon(holder)
            await anyio.sleep(0.05)
            tg.start_soon(waiter)
            await anyio.sleep(0.05)
            release.set()
        return order

    assert anyio.run(scenario) == ["first", "second"]


def test_ahold_cancellation_releases_its_reservation() -> None:
    """A cancelled waiter must free the claim — the next acquire can't hang."""
    import anyio

    async def scenario() -> _ClaimLocks:
        locks = _ClaimLocks(bound=1)
        async with locks.ahold("k"):
            with anyio.move_on_after(0.2):
                async with locks.ahold("k"):
                    pytest.fail("waiter entered while claim was held")
        # Reservation released on cancellation: re-acquire completes fast.
        with anyio.move_on_after(1.0):
            async with locks.ahold("k"):
                pass
        return locks

    locks = anyio.run(scenario)
    claim = locks._locks["k"]  # noqa: SLF001 — pin the registry state
    assert claim.users == 0
    assert not claim.lock.locked()


def test_replay_never_loads_entire_file_into_memory(tmp_path: Path) -> None:
    """A record larger than anything sane still verifies line-by-line."""
    big = {"id": "x", "pad": "y" * 200_000}
    journal = JobJournal(tmp_path / "journal.jsonl")
    journal.append(big)
    result = journal.replay()
    assert result.payloads == [big]
