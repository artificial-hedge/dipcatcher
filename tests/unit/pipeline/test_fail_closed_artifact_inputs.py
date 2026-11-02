"""Fail-closed artifact identity: clocks, counts, and manifest bytes.

SYNTHETIC panels only. No research score is claimed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from quant_fund.pipeline.artifact_manifest import (
    ArtifactManifestError,
    DatasetIdentity,
    DatasetIdentityError,
    build_dataset_identity,
    manifest_path,
    verify_artifact_manifest,
)


def _frame(stamp: datetime, *, null_clock: bool = False) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "event_time": [stamp],
            "security_id": ["S"],
            "future_return_5": [0.01],
        }
    )
    if null_clock:
        zone = "UTC" if stamp.tzinfo is not None else None
        frame = frame.with_columns(pl.lit(None, dtype=pl.Datetime("us", zone)).alias("event_time"))
    return frame


def _identity(**overrides: object) -> DatasetIdentity:
    payload = {
        "materialized_panel_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "row_count": 1,
        "column_count": 1,
        "time_start": "2020-01-01T00:00:00+00:00",
        "time_end": "2020-01-02T00:00:00Z",
        "label": "y",
        "label_horizon_bars": 1,
        "data_source": "synthetic",
    }
    payload.update(overrides)
    return DatasetIdentity(**payload)  # type: ignore[arg-type]


def test_aware_identity_timestamps_parse() -> None:
    identity = _identity()
    assert identity.time_end == "2020-01-02T00:00:00Z"


def test_naive_identity_timestamp_is_refused() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _identity(time_start="2020-01-01T00:00:00")


def test_dataset_identity_refuses_naive_and_null_clocks(tmp_path: Path) -> None:
    naive = _frame(datetime(2020, 1, 1))
    with pytest.raises(DatasetIdentityError, match="timezone-aware"):
        build_dataset_identity(
            naive,
            data_root=tmp_path,
            data_source="synthetic",
            label="future_return_5",
            label_horizon_bars=5,
        )
    aware = _frame(datetime(2020, 1, 1, tzinfo=UTC), null_clock=True)
    with pytest.raises(DatasetIdentityError, match="null"):
        build_dataset_identity(
            aware,
            data_root=tmp_path,
            data_source="synthetic",
            label="future_return_5",
            label_horizon_bars=5,
        )


@pytest.mark.parametrize("horizon", [True, False, 0, -1, 1.5])
def test_dataset_identity_refuses_bad_horizon(horizon: object, tmp_path: Path) -> None:
    frame = _frame(datetime(2020, 1, 1, tzinfo=UTC))
    with pytest.raises(DatasetIdentityError, match="positive integer"):
        build_dataset_identity(
            frame,
            data_root=tmp_path,
            data_source="synthetic",
            label="future_return_5",
            label_horizon_bars=horizon,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("source", ["", "   ", None, 3])
def test_dataset_identity_refuses_blank_source(source: object, tmp_path: Path) -> None:
    frame = _frame(datetime(2020, 1, 1, tzinfo=UTC))
    with pytest.raises(DatasetIdentityError, match="data_source"):
        build_dataset_identity(
            frame,
            data_root=tmp_path,
            data_source=source,  # type: ignore[arg-type]
            label="future_return_5",
            label_horizon_bars=5,
        )


def test_manifest_bytes_fail_closed(tmp_path: Path) -> None:
    payload = tmp_path / "model.joblib"
    payload.write_bytes(b"payload")
    target = manifest_path(payload)
    target.write_bytes(b"\xff")
    with pytest.raises(ArtifactManifestError, match="unreadable"):
        verify_artifact_manifest(payload)
    target.write_text('{"schema": "model_artifact.v1", "schema": "other"}', encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="duplicate"):
        verify_artifact_manifest(payload)
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="not a JSON object"):
        verify_artifact_manifest(payload)
    target.write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="unreadable"):
        verify_artifact_manifest(payload)
