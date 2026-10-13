"""Crash-resume recovery tests for the ingestion journal.

SYNTHETIC batches only (correctness tests, never market evidence). Proves a
killed ingest resumes from its cursor with no lost or duplicate rows, that
re-ingest is idempotent, and that a corrupted journal fails closed.
"""

from __future__ import annotations

import json

import pytest

from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def test_crash_resume_no_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run1 = IngestCheckpoint(path)
    run1.ingest(Batch(batch_id="b1", rows=100))
    run1.ingest(Batch(batch_id="b2", rows=50))
    # crash: new process reloads the journal from disk
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
    # crash + resume: the same batch is still a duplicate after reload
    run2 = IngestCheckpoint(path)
    assert run2.ingest(Batch(batch_id="b1", rows=100)) == "duplicate"
    assert run2.total_rows == 100


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
    # silently tamper with a committed row count -> must fail closed on reload
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"] = 999
    path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    try:
        IngestCheckpoint(path)
    except IngestJournalError:
        pass
    else:
        raise AssertionError("expected IngestJournalError on tampered journal")


def _sealed_record(batch_id: object, rows: object, **extra: object) -> str:
    body = {"batch_id": batch_id, "rows": rows, **extra}
    return json.dumps(
        {**body, "sha256": hash_bytes(canonical_json_bytes(body))}, sort_keys=True
    )


def test_reingest_with_conflicting_row_count_fails_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(batch_id="b1", rows=100))
    original = path.read_bytes()

    with pytest.raises(IngestJournalError, match="already committed"):
        run.ingest(Batch(batch_id="b1", rows=101))
    assert path.read_bytes() == original
    assert run.resume()["total_rows"] == 100


@pytest.mark.parametrize("replayed_rows", [100, 999])
def test_replay_rejects_duplicate_batch_id_even_with_valid_hash(tmp_path, replayed_rows) -> None:
    path = tmp_path / "ingest.jsonl"
    IngestCheckpoint(path).ingest(Batch(batch_id="b1", rows=100))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_sealed_record("b1", replayed_rows) + "\n")

    with pytest.raises(IngestJournalError, match="duplicate batch_id"):
        IngestCheckpoint(path)


@pytest.mark.parametrize(
    ("batch_id", "rows"),
    [
        ("", 1),
        ("   ", 1),
        (None, 1),
        ("b1", -1),
        ("b1", True),
        ("b1", 1.5),
        ("b1", "1"),
    ],
)
def test_invalid_batch_cannot_commit_or_replay(tmp_path, batch_id, rows) -> None:
    path = tmp_path / "ingest.jsonl"
    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path).ingest(Batch(batch_id=batch_id, rows=rows))
    assert not path.exists()

    path.write_text(_sealed_record(batch_id, rows) + "\n", encoding="utf-8")
    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)


def test_replay_rejects_extra_hash_valid_fields_and_blank_lines(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_text(_sealed_record("b1", 1, ignored=999) + "\n", encoding="utf-8")
    with pytest.raises(IngestJournalError, match="invalid fields"):
        IngestCheckpoint(path)

    path.write_text(_sealed_record("b1", 1) + "\n\n", encoding="utf-8")
    with pytest.raises(IngestJournalError, match="blank ingest journal line"):
        IngestCheckpoint(path)
