"""Crash-resume recovery tests for the ingestion pipeline cursor.

SYNTHETIC batches only (correctness tests, never market evidence). Proves a
crash mid-ingest resumes from the durable cursor with no duplicate rows and
idempotent re-ingest of a committed batch.
"""

from __future__ import annotations

from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError


def test_crash_resume_no_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run1 = IngestCheckpoint(path)
    assert run1.ingest(Batch(1, 100)) == "committed"
    assert run1.ingest(Batch(2, 50)) == "committed"
    # crash: drop the in-memory checkpoint, reload from disk.
    run2 = IngestCheckpoint(path)
    state = run2.resume()
    assert state["cursor"] == 2
    assert state["total_rows"] == 150
    assert state["n_batches"] == 2
    # continue after resume
    assert run2.ingest(Batch(3, 25)) == "committed"
    assert run2.total_rows == 175


def test_reingest_is_idempotent(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(1, 100))
    assert run.ingest(Batch(1, 100)) == "duplicate"
    assert run.total_rows == 100
    # even after crash, replay is a no-op
    run2 = IngestCheckpoint(path)
    assert run2.ingest(Batch(1, 100)) == "duplicate"
    assert run2.total_rows == 100


def test_resume_state_is_research_only(tmp_path) -> None:
    run = IngestCheckpoint(tmp_path / "ingest.jsonl")
    state = run.resume()
    assert state["research_only"] is True
    assert state["live_pnl_claim"] is False


def test_corrupt_journal_fails_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(1, 100))
    # tamper with a committed line -> self-hash mismatch on reload
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"rows":100', '"rows":999')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        IngestCheckpoint(path)
    except IngestJournalError:
        pass
    else:
        raise AssertionError("expected IngestJournalError on tampered journal")
