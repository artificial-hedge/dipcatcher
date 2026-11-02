"""Error-injection tests for ingestion journals and data manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_fund.data.ingest import data_manifest_contract_errors
from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _sealed_record(body: dict[str, Any]) -> str:
    sealed = {**body, "sha256": hash_bytes(canonical_json_bytes(body))}
    return json.dumps(sealed, sort_keys=True)


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "synthetic",
        "artifacts": {
            "bars": {
                "path": "/data/bars.parquet",
                "sha256": "a" * 64,
                "rows": 3,
                "columns": ["close", "event_time"],
            }
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"batch_id":"b1","rows":1',
        "{not json}",
        "[]",
        '"b1"',
        "null",
    ],
)
def test_malformed_journal_lines_raise_domain_error(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")

    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)


def test_truncated_suffix_does_not_return_partial_recovery(tmp_path: Path) -> None:
    path = tmp_path / "ingest.jsonl"
    committed = _sealed_record({"batch_id": "b1", "rows": 100})
    truncated = _sealed_record({"batch_id": "b2", "rows": 50})[:-9]
    path.write_text(committed + "\n" + truncated, encoding="utf-8")

    with pytest.raises(IngestJournalError, match="corrupt ingest journal line"):
        IngestCheckpoint(path)


def test_non_utf8_journal_raises_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_bytes(b"\xff\n")

    with pytest.raises(IngestJournalError, match="not valid UTF-8"):
        IngestCheckpoint(path)


@pytest.mark.parametrize("field", ["batch_id", "rows", "sha256"])
def test_tampered_sealed_fields_fail_closed(tmp_path: Path, field: str) -> None:
    path = tmp_path / "ingest.jsonl"
    record = json.loads(_sealed_record({"batch_id": "b1", "rows": 100}))
    if field == "batch_id":
        record[field] = "b2"
    elif field == "rows":
        record[field] = 101
    else:
        record[field] = "0" * 64
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(IngestJournalError, match="hash mismatch"):
        IngestCheckpoint(path)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"batch_id": "b1"},
        {"rows": 1},
        {"batch_id": "b1", "rows": 1, "ignored": "corruption"},
    ],
)
def test_hash_valid_records_require_exact_fields(tmp_path: Path, body: dict[str, Any]) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_text(_sealed_record(body) + "\n", encoding="utf-8")

    with pytest.raises(IngestJournalError, match="invalid fields"):
        IngestCheckpoint(path)


@pytest.mark.parametrize(
    ("batch_id", "rows"),
    [
        ("", 1),
        ("   ", 1),
        (None, 1),
        (7, 1),
        ([], 1),
        ("b1", -1),
        ("b1", True),
        ("b1", 1.5),
        ("b1", "1"),
        ("b1", None),
        ("b1", []),
    ],
)
def test_hash_valid_records_reject_invalid_batch_values(
    tmp_path: Path, batch_id: object, rows: object
) -> None:
    path = tmp_path / "ingest.jsonl"
    body = {"batch_id": batch_id, "rows": rows}
    path.write_text(_sealed_record(body) + "\n", encoding="utf-8")

    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path)


@pytest.mark.parametrize("blank_suffix", ["\n", "   \n"])
def test_blank_journal_lines_fail_closed(tmp_path: Path, blank_suffix: str) -> None:
    path = tmp_path / "ingest.jsonl"
    committed = _sealed_record({"batch_id": "b1", "rows": 1})
    path.write_text(committed + "\n" + blank_suffix, encoding="utf-8")

    with pytest.raises(IngestJournalError, match="blank ingest journal line"):
        IngestCheckpoint(path)


@pytest.mark.parametrize("replayed_rows", [100, 999])
def test_repeated_journal_batch_id_fails_closed(tmp_path: Path, replayed_rows: int) -> None:
    path = tmp_path / "ingest.jsonl"
    first = _sealed_record({"batch_id": "b1", "rows": 100})
    replay = _sealed_record({"batch_id": "b1", "rows": replayed_rows})
    path.write_text(first + "\n" + replay + "\n", encoding="utf-8")

    with pytest.raises(IngestJournalError, match="duplicate batch_id"):
        IngestCheckpoint(path)


@pytest.mark.parametrize(
    ("batch_id", "rows"),
    [("", 1), (None, 1), ("b1", -1), ("b1", True), ("b1", 1.5), ("b1", "1")],
)
def test_invalid_batch_cannot_commit(tmp_path: Path, batch_id: object, rows: object) -> None:
    path = tmp_path / "ingest.jsonl"

    with pytest.raises(IngestJournalError):
        IngestCheckpoint(path).ingest(Batch(batch_id=batch_id, rows=rows))
    assert not path.exists()


def test_conflicting_replay_does_not_mutate_journal_or_cursor(tmp_path: Path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(batch_id="b1", rows=100))
    original = path.read_bytes()

    with pytest.raises(IngestJournalError, match="already committed"):
        run.ingest(Batch(batch_id="b1", rows=101))
    assert path.read_bytes() == original
    assert run.resume()["total_rows"] == 100


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
def test_manifest_rejects_non_integer_schema_version(schema_version: object) -> None:
    manifest = _valid_manifest()
    manifest["schema_version"] = schema_version

    assert "schema_version_not_1" in data_manifest_contract_errors(manifest)


@pytest.mark.parametrize("source", [1, True, [], {}, "", "   ", None])
def test_manifest_rejects_missing_or_non_string_source(source: object) -> None:
    manifest = _valid_manifest()
    manifest["source"] = source

    assert "source_missing" in data_manifest_contract_errors(manifest)


@pytest.mark.parametrize("path_value", [None, 7, "", "   ", []])
def test_manifest_rejects_invalid_artifact_path(path_value: object) -> None:
    manifest = _valid_manifest()
    manifest["artifacts"]["bars"]["path"] = path_value

    assert "artifact:bars:path_invalid" in data_manifest_contract_errors(manifest)


def test_manifest_rejects_missing_artifact_path() -> None:
    manifest = _valid_manifest()
    del manifest["artifacts"]["bars"]["path"]

    assert data_manifest_contract_errors(manifest) == ["artifact:bars:path_invalid"]


@pytest.mark.parametrize("name", ["", "   ", 7, None])
def test_manifest_rejects_invalid_artifact_name(name: object) -> None:
    manifest = _valid_manifest()
    manifest["artifacts"] = {name: manifest["artifacts"]["bars"]}

    errors = data_manifest_contract_errors(manifest)
    assert f"artifact:{name}:name_invalid" in errors


def test_valid_manifest_contract_control() -> None:
    assert data_manifest_contract_errors(_valid_manifest()) == []
