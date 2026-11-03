"""Fail-closed contract checks on the sealed data manifest and provider routing.

``data_manifest_contract_errors`` is the verify-receipt gate: a manifest that
names a non-hex digest, a negative row count, a missing path, or unsorted
columns must be reported, never repaired. No network calls here.
"""

from __future__ import annotations

import copy
import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.hf_ohlcv_1m import HfOhlcv1mProvider
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.ingest import (
    PublicMarketProvider,
    data_manifest_contract_errors,
    make_provider,
)

_DIGEST = "a" * 64


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "synthetic",
        "artifacts": {
            "bars": {
                "path": "/lake/bronze/bars.parquet",
                "sha256": "ab" * 32,
                "rows": 3,
                "columns": ["close", "open"],
            }
        },
        "receipt_sha256": "cd" * 32,
    }


def _cfg(tmp_path: Path, **data: object) -> AppConfig:
    return AppConfig.model_validate({"data": {"root": str(tmp_path), **data}})


def test_a_sealed_manifest_passes_the_contract() -> None:
    assert data_manifest_contract_errors(_manifest()) == []


@pytest.mark.parametrize("schema_version", [0, 2, None, "1"])
def test_schema_version_must_be_exactly_one(schema_version: object) -> None:
    manifest = _manifest()
    manifest["schema_version"] = schema_version
    assert data_manifest_contract_errors(manifest) == ["schema_version_not_1"]


def test_schema_version_bool_is_not_one() -> None:
    """A bool is not schema version 1, even though ``True == 1``."""
    manifest = _manifest()
    manifest["schema_version"] = True
    assert data_manifest_contract_errors(manifest) == ["schema_version_not_1"]


@pytest.mark.parametrize("source", ["", None, 0])
def test_source_must_be_present(source: object) -> None:
    manifest = _manifest()
    manifest["source"] = source
    assert data_manifest_contract_errors(manifest) == ["source_missing"]


@pytest.mark.parametrize("artifacts", [None, {}, [], "bars"])
def test_missing_artifacts_short_circuits(artifacts: object) -> None:
    manifest = _manifest()
    manifest["artifacts"] = artifacts
    assert data_manifest_contract_errors(manifest) == ["artifacts_missing"]


def test_missing_artifacts_key_short_circuits() -> None:
    manifest = _manifest()
    manifest.pop("artifacts")
    assert data_manifest_contract_errors(manifest) == ["artifacts_missing"]


def test_non_object_artifact_is_reported_without_crashing() -> None:
    manifest = _manifest()
    manifest["artifacts"]["silver"] = ["not", "an", "object"]
    assert data_manifest_contract_errors(manifest) == ["artifact:silver:not_object"]


@pytest.mark.parametrize("digest", ["", "AB" * 32, "ab" * 31, "ab" * 33, "g" * 64, None, 5])
def test_bad_digest_is_reported(digest: object) -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["sha256"] = digest
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:sha256_invalid"]


@pytest.mark.parametrize("rows", [-1, True, False, 1.0, "3", None])
def test_bad_row_count_is_reported(rows: object) -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["rows"] = rows
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:rows_invalid"]


def test_zero_rows_is_a_valid_empty_artifact() -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["rows"] = 0
    assert data_manifest_contract_errors(manifest) == []


@pytest.mark.parametrize(
    "columns",
    [[], ["open", "close"], ["close", "open", 1], "open", None, ("a", "b"), {}],
)
def test_bad_column_list_is_reported(columns: object) -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["columns"] = columns
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:columns_invalid"]


def test_duplicate_column_names_are_tolerated_by_the_contract() -> None:
    """The contract pins sort order only; uniqueness is not enforced here."""
    manifest = _manifest()
    manifest["artifacts"]["bars"]["columns"] = ["a", "a"]
    assert data_manifest_contract_errors(manifest) == []


def test_every_artifact_is_reported_not_just_the_first() -> None:
    manifest = _manifest()
    manifest["artifacts"]["silver"] = {"sha256": "nope", "rows": -2, "columns": []}
    errors = data_manifest_contract_errors(manifest)
    assert errors == [
        "artifact:silver:path_invalid",
        "artifact:silver:sha256_invalid",
        "artifact:silver:rows_invalid",
        "artifact:silver:columns_invalid",
    ]


def test_all_problems_are_collected_not_short_circuited() -> None:
    errors = data_manifest_contract_errors(
        {
            "schema_version": 2,
            "source": "",
            "artifacts": {
                "bars": {"sha256": "bad", "rows": -1, "columns": ["b", "a"]},
                "silver": {"sha256": _DIGEST, "rows": 0, "columns": ["a"]},
            },
        }
    )
    assert errors == [
        "schema_version_not_1",
        "source_missing",
        "artifact:bars:path_invalid",
        "artifact:bars:sha256_invalid",
        "artifact:bars:rows_invalid",
        "artifact:bars:columns_invalid",
        "artifact:silver:path_invalid",
    ]


def test_missing_digest_key_reports_the_digest() -> None:
    manifest = _manifest()
    del manifest["artifacts"]["bars"]["sha256"]
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:sha256_invalid"]


def test_contract_errors_do_not_mutate_the_manifest() -> None:
    manifest = _manifest()
    before = copy.deepcopy(manifest)
    data_manifest_contract_errors(manifest)
    assert manifest == before


def test_public_provider_requires_a_path_for_file_backed_tapes(tmp_path: Path) -> None:
    for source in ("nasdaq_itch", "fi_2010"):
        cfg = _cfg(tmp_path, source=source)
        with pytest.raises(ValueError, match="requires data.source_path"):
            PublicMarketProvider(cfg)


def test_registry_failure_is_wrapped_with_the_source_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_module = importlib.import_module("quant_fund.data.ingest")

    def boom(name: str) -> object:
        raise ValueError(f"unknown public data source {name!r}")

    monkeypatch.setattr(ingest_module, "get_source", boom)
    with pytest.raises(ValueError, match="cannot configure public source 'gdelt'"):
        PublicMarketProvider(_cfg(tmp_path, source="gdelt"))


def test_registry_type_error_is_wrapped_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_module = importlib.import_module("quant_fund.data.ingest")

    monkeypatch.setattr(
        ingest_module, "get_source", lambda name: (_ for _ in ()).throw(TypeError("no ctor"))
    )
    with pytest.raises(ValueError, match="cannot configure public source 'sec_edgar'"):
        PublicMarketProvider(_cfg(tmp_path, source="sec_edgar"))


def test_non_bar_source_refuses_to_produce_bars(tmp_path: Path) -> None:
    provider = PublicMarketProvider(_cfg(tmp_path, source="fred"))
    with pytest.raises(ValueError, match="is not a bar provider"):
        provider.get_bars()
    assert provider.get_corporate_actions().is_empty()
    assert provider.get_security_master().is_empty()


def test_make_provider_rejects_registered_non_bar_sources(tmp_path: Path) -> None:
    for source in ("fred", "gdelt", "bea"):
        with pytest.raises(ValueError, match="not a market-bar provider"):
            make_provider(_cfg(tmp_path, source=source))


def test_make_provider_routes_hf_cache_offline(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, source="hf_ohlcv_1m", source_symbol="MSFT", source_interval="1m")
    provider = make_provider(cfg)
    assert isinstance(provider, HfOhlcv1mProvider)
    assert provider.allow_download is False
    assert provider.max_months == 24
    assert provider.cache_dir == tmp_path / "hf_ohlcv_1m"
    assert list(provider.symbols) == ["MSFT"]


def test_make_provider_normalizes_whitespace_and_case(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, source="synthetic")
    object.__setattr__(cfg.data, "source", "  SYNTHETIC  ")
    assert isinstance(make_provider(cfg), SyntheticMarketProvider)
    object.__setattr__(cfg.data, "source", " FILE ")
    assert isinstance(make_provider(cfg), ParquetMarketProvider)


def test_public_bars_respect_start_end_and_security_filter(tmp_path: Path) -> None:
    csv_path = tmp_path / "itch.csv"
    csv_path.write_text(
        "security_id,event_time,open,high,low,close,volume\n"
        "A,2024-01-02T00:00:00Z,10,11,9,10.5,100\n"
        "A,2024-01-03T00:00:00Z,11,12,10,11.5,100\n"
        "B,2024-01-03T00:00:00Z,20,21,19,20.5,100\n"
    )
    provider = PublicMarketProvider(_cfg(tmp_path, source="nasdaq_itch", source_path=str(csv_path)))
    assert provider.get_bars().height == 3

    start = datetime(2024, 1, 3, tzinfo=UTC)
    assert provider.get_bars(start=start).height == 2
    assert provider.get_bars(start=start, security_ids=["B"]).height == 1
    end = datetime(2024, 1, 2, 23, tzinfo=UTC)
    assert provider.get_bars(end=end)["security_id"].to_list() == ["A"]

    future = datetime(2030, 1, 1, tzinfo=UTC)
    assert provider.get_bars(start=future).is_empty()
    assert provider.get_bars(end=datetime(1990, 1, 1, tzinfo=UTC)).is_empty()
    far = datetime(2024, 1, 3, tzinfo=UTC) + timedelta(days=3650)
    assert provider.get_bars(start=far).is_empty()
    assert (
        provider.get_bars(
            security_ids=["A"], end=datetime(2024, 1, 2, tzinfo=UTC) + timedelta(hours=1)
        ).height
        == 1
    )
