"""SYNTHETIC corruption/fault tests for the journal's byte-level contract."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fx1.serve.journal import JobJournal, _line_bytes

ZERO_CHAIN = "0" * 64


def _record(seq: Any, payload: Any, chain: str = ZERO_CHAIN) -> bytes:
    """Independently encode even invalid shapes with a matching digest."""
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


@pytest.mark.parametrize(
    "payload", [{}, {"a": 1}, {"text": "café\n東京"}, {"x": [None, True, 1.25]}]
)
def test_canonical_bytes_are_unchanged(payload: dict[str, Any]) -> None:
    assert _line_bytes(0, ZERO_CHAIN, payload) == _record(0, payload)


@pytest.mark.parametrize("payload", [None, [], "not an object", 7, True])
def test_replay_rejects_non_object_payloads(tmp_path: Path, payload: Any) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(_record(0, payload))
    result = JobJournal(path).replay()
    assert result.payloads == []
    assert result.truncated_at == 0
    assert result.dropped == 1


@pytest.mark.parametrize("seq", [False, 0.0])
def test_replay_requires_an_integer_sequence(tmp_path: Path, seq: Any) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(_record(seq, {"id": "synthetic"}))
    result = JobJournal(path).replay()
    assert result.payloads == []
    assert result.truncated_at == 0


def test_missing_newline_is_a_torn_tail(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    raw = _record(0, {"id": "synthetic"})[:-1]
    path.write_bytes(raw)
    journal = JobJournal(path)
    result = journal.replay()
    assert result.truncated_at == 0
    assert result.payloads == []
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "unreachable"})
    assert path.read_bytes() == raw


def test_damaged_tail_blocks_append_without_rewriting_evidence(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    first = _record(0, {"id": "first"})
    raw = first + b'{"torn":\n' + b"garbage\n" + b"last"
    path.write_bytes(raw)
    journal = JobJournal(path)
    result = journal.replay()
    assert result.payloads == [{"id": "first"}]
    assert result.truncated_at == len(first)
    assert result.dropped == 3
    assert len(result.warnings) == 1
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "unreachable"})
    assert path.read_bytes() == raw


def test_explicit_compaction_restores_writes_after_corruption(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(_record(0, {"id": "first"}) + b"broken")
    journal = JobJournal(path)
    result = journal.replay()
    journal.compact(result.payloads)
    journal.append({"id": "second"})
    assert JobJournal(path).replay().payloads == [{"id": "first"}, {"id": "second"}]


def test_failed_fsync_blocks_retry_until_replay(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    with (
        patch("fx1.serve.journal.os.fsync", side_effect=OSError("synthetic fsync failure")),
        pytest.raises(OSError, match="synthetic fsync"),
    ):
        journal.append({"id": "uncertain"})
    after_failure = path.read_bytes()
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "must not reuse sequence zero"})
    assert path.read_bytes() == after_failure
    assert journal.replay().payloads == [{"id": "uncertain"}]
    journal.append({"id": "second"})
    result = JobJournal(path).replay()
    assert result.truncated_at is None
    assert result.payloads == [{"id": "uncertain"}, {"id": "second"}]


def test_new_journal_fsyncs_parent_before_acknowledging_append(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    with (
        patch(
            "fx1.serve.journal._fsync_parent", side_effect=OSError("synthetic directory fsync")
        ) as sync_parent,
        pytest.raises(OSError, match="synthetic directory fsync"),
    ):
        journal.append({"id": "uncertain"})
    sync_parent.assert_called_once_with(tmp_path)
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "must not reuse sequence zero"})
    assert journal.replay().payloads == [{"id": "uncertain"}]
    journal.append({"id": "second"})
    assert JobJournal(path).replay().payloads == [
        {"id": "uncertain"},
        {"id": "second"},
    ]


def test_existing_journal_append_does_not_resync_parent(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": 0})
    with patch("fx1.serve.journal._fsync_parent") as sync_parent:
        journal.append({"id": 1})
    sync_parent.assert_not_called()


def test_replay_streams_instead_of_reading_the_whole_file(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "journal.jsonl")
    journal.compact([{"id": i} for i in range(100)])
    with patch.object(Path, "read_bytes", side_effect=AssertionError("unbounded read")):
        result = journal.replay()
    assert result.payloads == [{"id": i} for i in range(100)]


def test_programming_errors_are_not_silenced_as_corruption(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "journal.jsonl")
    journal.append({"id": 0})
    with (
        patch("fx1.serve.journal.json.loads", side_effect=RuntimeError("synthetic decoder bug")),
        pytest.raises(RuntimeError, match="synthetic decoder bug"),
    ):
        journal.replay()
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": 1})
    assert journal.replay().payloads == [{"id": 0}]


@pytest.mark.parametrize("raw", [b"[]\n", b"null\n", b"{}\n", b"\xff\n", b'{"seq":\n'])
def test_malformed_records_are_reported_without_mutation(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(raw)
    result = JobJournal(path).replay()
    assert result.truncated_at == 0
    assert result.dropped == 1
    assert path.read_bytes() == raw


def test_replay_resets_chain_when_file_was_removed(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": "old"})
    path.unlink()
    assert journal.replay().payloads == []
    assert not path.exists()
    journal.append({"id": "new"})
    result = JobJournal(path).replay()
    assert result.truncated_at is None
    assert result.payloads == [{"id": "new"}]


def test_successful_replay_restores_next_sequence(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": 0})
    journal.append({"id": 1})
    recovered = JobJournal(path)
    assert recovered.replay().payloads == [{"id": 0}, {"id": 1}]
    recovered.append({"id": 2})
    result = JobJournal(path).replay()
    assert result.truncated_at is None
    assert result.payloads == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_serialization_error_does_not_block_a_clean_journal(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "journal.jsonl")
    with pytest.raises(TypeError):
        journal.append({"bad": object()})
    journal.append({"id": "valid"})
    assert journal.replay().payloads == [{"id": "valid"}]


def test_concurrent_appends_keep_a_single_verified_chain(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: journal.append({"id": i}), range(80)))
    result = JobJournal(path).replay()
    assert result.truncated_at is None
    assert sorted(row["id"] for row in result.payloads) == list(range(80))


@pytest.mark.parametrize("field,value", [("seq", 3), ("chain", "bad"), ("sha256", "bad")])
def test_corrupt_metadata_stops_at_the_exact_record(tmp_path: Path, field: str, value: Any) -> None:
    first = _record(0, {"id": 0})
    chain = hashlib.sha256(first).hexdigest()
    second = json.loads(_record(1, {"id": 1}, chain))
    second[field] = value
    damaged = json.dumps(second).encode() + b"\n"
    path = tmp_path / "journal.jsonl"
    path.write_bytes(first + damaged)
    result = JobJournal(path).replay()
    assert result.payloads == [{"id": 0}]
    assert result.truncated_at == len(first)
    assert result.dropped == 1


def test_legacy_crlf_records_keep_their_raw_chain(tmp_path: Path) -> None:
    first = _record(0, {"id": 0})[:-1] + b"\r\n"
    second = _record(1, {"id": 1}, hashlib.sha256(first).hexdigest())
    path = tmp_path / "journal.jsonl"
    path.write_bytes(first + second)
    journal = JobJournal(path)
    assert journal.replay().payloads == [{"id": 0}, {"id": 1}]
    journal.append({"id": 2})
    assert JobJournal(path).replay().payloads == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_clean_external_repair_unblocks_the_same_instance(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    valid = _record(0, {"id": 0})
    path.write_bytes(valid + b"bad\n")
    journal = JobJournal(path)
    assert journal.replay().truncated_at == len(valid)
    path.write_bytes(valid)
    assert journal.replay().truncated_at is None
    journal.append({"id": 1})
    assert JobJournal(path).replay().payloads == [{"id": 0}, {"id": 1}]


def test_compaction_failure_keeps_the_old_chain_and_payloads(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": 0})
    original = path.read_bytes()
    with (
        patch("fx1.serve.journal.os.replace", side_effect=OSError("synthetic replace failure")),
        pytest.raises(OSError, match="synthetic replace"),
    ):
        journal.compact([{"id": "uncommitted"}])
    assert path.read_bytes() == original
    journal.append({"id": 1})
    assert JobJournal(path).replay().payloads == [{"id": 0}, {"id": 1}]


def test_compaction_fsyncs_parent_and_fails_closed_on_error(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": "old"})
    with (
        patch(
            "fx1.serve.journal._fsync_parent", side_effect=OSError("synthetic directory fsync")
        ) as sync_parent,
        pytest.raises(OSError, match="synthetic directory fsync"),
    ):
        journal.compact([{"id": "replacement"}])
    sync_parent.assert_called_once_with(tmp_path)
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": "must replay replacement"})
    assert journal.replay().payloads == [{"id": "replacement"}]
    journal.append({"id": "after recovery"})
    assert JobJournal(path).replay().payloads == [
        {"id": "replacement"},
        {"id": "after recovery"},
    ]


def test_read_failure_blocks_writes_until_successful_recovery(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = JobJournal(path)
    journal.append({"id": 0})
    with (
        patch.object(Path, "open", side_effect=PermissionError("synthetic unreadable journal")),
        pytest.raises(PermissionError, match="synthetic unreadable"),
    ):
        journal.replay()
    with pytest.raises(RuntimeError, match="replay|compact"):
        journal.append({"id": 1})
    assert journal.replay().payloads == [{"id": 0}]
    journal.append({"id": 1})
    assert JobJournal(path).replay().payloads == [{"id": 0}, {"id": 1}]


def test_decoder_depth_limit_is_reported_as_corruption(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b"[" * 10000 + b"0" + b"]" * 10000 + b"\n")
    result = JobJournal(path).replay()
    assert result.payloads == []
    assert result.truncated_at == 0
    assert result.dropped == 1


def test_valid_prefix_matches_the_legacy_wire_encoder(tmp_path: Path) -> None:
    payloads = [
        {"id": i, "nested": {"null": None, "flag": i % 2 == 0, "value": i / 7}, "tags": ["é", i]}
        for i in range(120)
    ]
    chain = ZERO_CHAIN
    raw_lines = []
    for seq, payload in enumerate(payloads):
        raw = _record(seq, payload, chain)
        raw_lines.append(raw)
        chain = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "journal.jsonl"
    original = b"".join(raw_lines)
    path.write_bytes(original)
    journal = JobJournal(path)
    assert journal.replay().payloads == payloads
    assert path.read_bytes() == original
    journal.compact(payloads)
    assert path.read_bytes() == original
