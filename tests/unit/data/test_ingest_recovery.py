"""Crash-resume recovery tests for the ingestion journal.

SYNTHETIC batches only (correctness tests, never market evidence). Proves a
killed ingest resumes from its cursor with no lost or duplicate rows, that
re-ingest is idempotent, and that a corrupted journal fails closed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _record(batch: Batch, **extra: Any) -> bytes:
    rec: dict[str, Any] = {"batch_id": batch.batch_id, "rows": batch.rows, **extra}
    rec["sha256"] = hash_bytes(canonical_json_bytes(rec))
    return json.dumps(rec, sort_keys=True).encode("utf-8")


def test_crash_resume_no_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run1 = IngestCheckpoint(path)
    run1.ingest(Batch(batch_id="b1", rows=100))
    run1.ingest(Batch(batch_id="b2", rows=50))
    run2 = IngestCheckpoint(path)
    assert run2.cursor == 2
    assert run2.total_rows == 150
    assert run2.n_batches == 2
    assert run2.ingest(Batch(batch_id="b3", rows=25)) == "committed"
    assert run2.total_rows == 175


def test_reingest_is_idempotent(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    assert run.ingest(Batch(batch_id="b1", rows=100)) == "committed"
    assert run.ingest(Batch(batch_id="b1", rows=100)) == "duplicate"
    assert run.total_rows == 100
    run2 = IngestCheckpoint(path)
    assert run2.ingest(Batch(batch_id="b1", rows=100)) == "duplicate"
    assert run2.total_rows == 100


def test_conflicting_reingest_fails_without_mutation(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(batch_id="b1", rows=100))
    before = path.read_bytes()
    with pytest.raises(IngestJournalError, match="already committed"):
        run.ingest(Batch(batch_id="b1", rows=101))
    assert path.read_bytes() == before
    assert run.resume()["total_rows"] == 100


def test_resume_state_is_research_only(tmp_path) -> None:
    run = IngestCheckpoint(tmp_path / "ingest.jsonl")
    run.ingest(Batch(batch_id="b1", rows=1))
    state = run.resume()
    assert state["research_only"] is True
    assert state["live_pnl_claim"] is False


def test_corrupt_journal_fails_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(batch_id="b1", rows=100))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"] = 999
    path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)


@pytest.mark.parametrize("cut", [1, 17, 40, -1])
def test_torn_final_record_is_discarded_and_replayed(tmp_path, cut: int) -> None:
    path = tmp_path / "ingest.jsonl"
    first = _record(Batch(batch_id="b1", rows=100)) + b"\n"
    second = _record(Batch(batch_id="b2", rows=50))
    path.write_bytes(first + second[:cut])
    run = IngestCheckpoint(path)
    assert run.resume()["total_rows"] == 100
    assert path.read_bytes() == first
    assert run.ingest(Batch(batch_id="b2", rows=50)) == "committed"
    resumed = IngestCheckpoint(path)
    assert resumed.resume()["total_rows"] == 150
    assert resumed.ingest(Batch(batch_id="b2", rows=50)) == "duplicate"


def test_complete_final_record_without_newline_is_repaired(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    first = _record(Batch(batch_id="b1", rows=100)) + b"\n"
    second = _record(Batch(batch_id="b2", rows=50))
    path.write_bytes(first + second)
    run = IngestCheckpoint(path)
    assert run.resume()["total_rows"] == 150
    assert path.read_bytes() == first + second + b"\n"
    assert run.ingest(Batch(batch_id="b3", rows=25)) == "committed"
    assert IngestCheckpoint(path).resume()["total_rows"] == 175


@pytest.mark.parametrize("terminated", [False, True])
def test_hash_invalid_final_record_is_never_discarded(tmp_path, terminated: bool) -> None:
    path = tmp_path / "ingest.jsonl"
    valid = _record(Batch(batch_id="b1", rows=100)) + b"\n"
    invalid = json.loads(_record(Batch(batch_id="b2", rows=50)))
    invalid["sha256"] = "0" * 64
    raw = valid + json.dumps(invalid, sort_keys=True).encode("utf-8")
    raw += b"\n" if terminated else b""
    path.write_bytes(raw)
    with pytest.raises(IngestJournalError, match="hash mismatch"):
        IngestCheckpoint(path)
    assert path.read_bytes() == raw


def test_malformed_terminated_record_is_not_repaired(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    raw = _record(Batch(batch_id="b1", rows=100)) + b'\n{"batch_id":\n'
    path.write_bytes(raw)
    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)
    assert path.read_bytes() == raw


def test_duplicate_batch_ids_on_replay_fail_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    record = _record(Batch(batch_id="b1", rows=100))
    raw = record + b"\n" + record + b"\n"
    path.write_bytes(raw)
    with pytest.raises(IngestJournalError, match="duplicate batch_id"):
        IngestCheckpoint(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("raw", [b"\n", b'{"batch_id":"b1","rows":1}\n'])
def test_blank_or_unsealed_records_fail_closed(tmp_path, raw: bytes) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_bytes(raw)
    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)
    assert path.read_bytes() == raw


def test_extra_record_fields_fail_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_bytes(_record(Batch(batch_id="b1", rows=1), unexpected=True) + b"\n")
    with pytest.raises(IngestJournalError, match="invalid fields"):
        IngestCheckpoint(path)


@pytest.mark.parametrize("batch_id", ["", "   ", None, 7])
def test_invalid_batch_id_is_rejected(tmp_path, batch_id: Any) -> None:
    with pytest.raises(IngestJournalError, match="batch_id"):
        IngestCheckpoint(tmp_path / "ingest.jsonl").ingest(Batch(batch_id, 1))


@pytest.mark.parametrize("rows", [-1, True, 1.5, "1", None])
def test_invalid_row_count_is_rejected(tmp_path, rows: Any) -> None:
    with pytest.raises(IngestJournalError, match="rows"):
        IngestCheckpoint(tmp_path / "ingest.jsonl").ingest(Batch("b1", rows))
