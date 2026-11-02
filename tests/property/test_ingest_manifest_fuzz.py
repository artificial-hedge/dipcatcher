"""Bounded fuzzing of ingestion journal recovery and manifest mappings."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.data.ingest import data_manifest_contract_errors
from quant_fund.data.ingest_resume import IngestCheckpoint, IngestJournalError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=12)
_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-1_000, max_value=1_000)
    | st.floats(
        min_value=-1e6,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
    )
    | _TEXT
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(_TEXT, children, max_size=3),
    max_leaves=8,
)
_FUZZ_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    database=None,
)


def _valid_artifact() -> dict[str, Any]:
    return {
        "path": "/data/bars.parquet",
        "sha256": "a" * 64,
        "rows": 3,
        "columns": ["close", "event_time"],
    }


def _is_valid_manifest(manifest: dict[str, Any]) -> bool:
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        return False
    source = manifest.get("source")
    if not isinstance(source, str) or not source.strip():
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for name, item in artifacts.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            return False
        path = item.get("path")
        digest = item.get("sha256")
        rows = item.get("rows")
        columns = item.get("columns")
        if not isinstance(path, str) or not path.strip():
            return False
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            return False
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            return False
        if not (
            isinstance(columns, list)
            and columns
            and all(isinstance(column, str) for column in columns)
            and columns == sorted(columns)
        ):
            return False
    return True


_FUZZ_ARTIFACT = st.one_of(
    st.just(_valid_artifact()),
    _JSON_VALUE,
    st.fixed_dictionaries(
        {
            "path": st.one_of(st.just("/data/bars.parquet"), _JSON_VALUE),
            "sha256": st.one_of(st.just("a" * 64), _JSON_VALUE),
            "rows": st.one_of(st.just(3), _JSON_VALUE),
            "columns": st.one_of(
                st.just(["close", "event_time"]),
                st.lists(_JSON_VALUE, max_size=4),
            ),
        }
    ),
)
_FUZZ_MANIFEST = st.fixed_dictionaries(
    {
        "schema_version": st.one_of(st.just(1), _JSON_VALUE),
        "source": st.one_of(st.just("synthetic"), _JSON_VALUE),
        "artifacts": st.dictionaries(_TEXT, _FUZZ_ARTIFACT, max_size=3),
    }
)
_VALID_MANIFEST = st.fixed_dictionaries(
    {
        "schema_version": st.just(1),
        "source": st.sampled_from(["synthetic", "parquet", "hf_ohlcv_1m"]),
        "artifacts": st.dictionaries(
            st.text(alphabet="abcxyz_", min_size=1, max_size=8),
            st.fixed_dictionaries(
                {
                    "path": st.just("/data/bars.parquet"),
                    "sha256": st.binary(min_size=32, max_size=32).map(bytes.hex),
                    "rows": st.integers(min_value=0, max_value=1_000_000),
                    "columns": st.lists(
                        st.text(alphabet="abc_", min_size=1, max_size=5),
                        min_size=1,
                        max_size=4,
                    ).map(sorted),
                }
            ),
            min_size=1,
            max_size=3,
        ),
    }
)


@_FUZZ_SETTINGS
@given(manifest=_FUZZ_MANIFEST)
def test_manifest_never_silently_accepts_invalid_mapping(
    manifest: dict[str, Any],
) -> None:
    errors = data_manifest_contract_errors(manifest)

    assert all(isinstance(error, str) for error in errors)
    if not errors:
        assert _is_valid_manifest(manifest)


@_FUZZ_SETTINGS
@given(manifest=_VALID_MANIFEST)
def test_valid_manifest_mappings_pass(manifest: dict[str, Any]) -> None:
    assert _is_valid_manifest(manifest)
    assert data_manifest_contract_errors(manifest) == []


@_FUZZ_SETTINGS
@given(batch_id=_JSON_VALUE, rows=_JSON_VALUE)
def test_sealed_journal_values_pass_only_with_valid_batch_schema(batch_id: Any, rows: Any) -> None:
    body = {"batch_id": batch_id, "rows": rows}
    record = {**body, "sha256": hash_bytes(canonical_json_bytes(body))}
    valid = isinstance(batch_id, str) and bool(batch_id.strip()) and type(rows) is int and rows >= 0

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "ingest.jsonl"
        path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        if valid:
            checkpoint = IngestCheckpoint(path)
            assert checkpoint.resume()["cursor"] == 1
            assert checkpoint.total_rows == rows
        else:
            with pytest.raises(IngestJournalError):
                IngestCheckpoint(path)


@_FUZZ_SETTINGS
@given(raw=st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=96))
def test_arbitrary_journal_text_has_no_unhandled_parser_escape(raw: str) -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "ingest.jsonl"
        path.write_text(raw, encoding="utf-8")
        try:
            checkpoint = IngestCheckpoint(path)
        except IngestJournalError:
            return

        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert checkpoint.n_batches == len(records)
        for record in records:
            assert set(record) == {"batch_id", "rows", "sha256"}
            assert isinstance(record["batch_id"], str) and record["batch_id"].strip()
            assert type(record["rows"]) is int and record["rows"] >= 0
