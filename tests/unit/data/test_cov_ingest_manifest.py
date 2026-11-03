"""Fail-closed ingest config routing and sealed-manifest contract checks."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

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


def _cfg(tmp_path: Path, **data: object) -> AppConfig:
    return AppConfig.model_validate({"data": {"root": str(tmp_path), **data}})


def _manifest(**overrides: object) -> dict[str, object]:
    artifact = {"sha256": _DIGEST, "rows": 3, "columns": ["a", "b"]}
    base: dict[str, object] = {
        "schema_version": 1,
        "source": "binance_public_data",
        "artifacts": {"bars": artifact},
    }
    base.update(overrides)
    return base


# -- provider routing --


def test_source_pathed_sources_require_a_path(tmp_path: Path) -> None:
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

    # An empty selection stays on the filtering path without raising on a null column.
    future = datetime(2030, 1, 1, tzinfo=UTC)
    assert provider.get_bars(start=future).is_empty()
    assert provider.get_bars(end=datetime(1990, 1, 1, tzinfo=UTC)).is_empty()
    assert provider.get_bars(
        start=datetime(2024, 1, 3, tzinfo=UTC) + timedelta(days=3650)
    ).is_empty()
    assert (
        provider.get_bars(
            security_ids=["A"], end=datetime(2024, 1, 2, tzinfo=UTC) + timedelta(hours=1)
        ).height
        == 1
    )


# -- sealed manifest contract --


def test_valid_manifest_has_no_errors() -> None:
    assert data_manifest_contract_errors(_manifest()) == []


@pytest.mark.parametrize("schema_version", [0, 2, "1", None])
def test_schema_version_must_be_exactly_one(schema_version: object) -> None:
    errors = data_manifest_contract_errors(_manifest(schema_version=schema_version))
    assert "schema_version_not_1" in errors


@pytest.mark.parametrize("source", ["", None])
def test_missing_source_is_reported(source: object) -> None:
    assert "source_missing" in data_manifest_contract_errors(_manifest(source=source))


@pytest.mark.parametrize("artifacts", [None, {}, [], "bars"])
def test_missing_or_non_mapping_artifacts_return_early(artifacts: object) -> None:
    errors = data_manifest_contract_errors(_manifest(artifacts=artifacts))
    assert errors == ["artifacts_missing"]


def test_artifact_entry_must_be_an_object() -> None:
    errors = data_manifest_contract_errors(_manifest(artifacts={"bars": ["bad"]}))
    assert errors == ["artifact:bars:not_object"]


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "a" * 65, "g" * 64, 123, None])
def test_sha256_must_be_lowercase_hex_of_length_64(digest: object) -> None:
    errors = data_manifest_contract_errors(
        _manifest(artifacts={"bars": {"sha256": digest, "rows": 0, "columns": ["a"]}})
    )
    assert errors == ["artifact:bars:sha256_invalid"]


@pytest.mark.parametrize("rows", [-1, 1.0, "3", None, True, False])
def test_rows_must_be_a_non_negative_non_bool_int(rows: object) -> None:
    errors = data_manifest_contract_errors(
        _manifest(artifacts={"bars": {"sha256": _DIGEST, "rows": rows, "columns": ["a"]}})
    )
    assert errors == ["artifact:bars:rows_invalid"]


@pytest.mark.parametrize(
    "columns",
    [None, "a", [], ["b", "a"], ["a", 1], {}, ("a", "b")],
)
def test_columns_must_be_a_non_empty_sorted_list_of_strings(columns: object) -> None:
    errors = data_manifest_contract_errors(
        _manifest(artifacts={"bars": {"sha256": _DIGEST, "rows": 1, "columns": columns}})
    )
    assert errors == ["artifact:bars:columns_invalid"]


def test_duplicate_column_names_are_tolerated_by_the_contract() -> None:
    """The contract pins sort order only; uniqueness is not enforced here."""
    errors = data_manifest_contract_errors(
        _manifest(artifacts={"bars": {"sha256": _DIGEST, "rows": 0, "columns": ["a", "a"]}})
    )
    assert errors == []


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
        "artifact:bars:sha256_invalid",
        "artifact:bars:rows_invalid",
        "artifact:bars:columns_invalid",
    ]


def test_empty_digest_only_reports_the_digest() -> None:
    # Empty sha256 is invalid but rows/columns are still checked independently.
    errors = data_manifest_contract_errors(
        _manifest(artifacts={"bars": {"rows": 0, "columns": ["a"]}})
    )
    assert errors == ["artifact:bars:sha256_invalid"]
