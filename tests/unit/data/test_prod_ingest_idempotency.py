from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError


def test_reused_batch_id_with_different_row_count_fails_closed(tmp_path: Path) -> None:
    checkpoint = IngestCheckpoint(tmp_path / "ingest.jsonl")
    assert checkpoint.ingest(Batch("batch-1", rows=10)) == "committed"

    with pytest.raises(IngestJournalError, match="different row count"):
        checkpoint.ingest(Batch("batch-1", rows=11))

    assert checkpoint.total_rows == 10
    resumed = IngestCheckpoint(tmp_path / "ingest.jsonl")
    assert resumed.total_rows == 10
    assert resumed.n_batches == 1
