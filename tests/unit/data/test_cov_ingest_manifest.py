"""Fail-closed contract checks on the sealed data manifest and provider routing.

``data_manifest_contract_errors`` is the verify-receipt gate: a manifest that
names a non-hex digest, a negative row count, or unsorted columns must be
reported, never repaired. No network calls here.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import (
    PublicMarketProvider,
    data_manifest_contract_errors,
    make_provider,
)


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


def test_a_sealed_manifest_passes_the_contract() -> None:
    assert data_manifest_contract_errors(_manifest()) == []


@pytest.mark.parametrize("schema_version", [0, 2, None, "1"])
def test_schema_version_must_be_exactly_one(schema_version: object) -> None:
    manifest = _manifest()
    manifest["schema_version"] = schema_version
    assert data_manifest_contract_errors(manifest) == ["schema_version_not_1"]


def test_schema_version_bool_compares_equal_to_one() -> None:
    """Documented quirk: ``True == 1`` in Python, so a bool schema passes."""
    manifest = _manifest()
    manifest["schema_version"] = True
    assert data_manifest_contract_errors(manifest) == []


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


@pytest.mark.parametrize("digest", ["", "AB" * 32, "ab" * 31, "ab" * 33, None, 5])
def test_bad_digest_is_reported(digest: object) -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["sha256"] = digest
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:sha256_invalid"]


@pytest.mark.parametrize("rows", [-1, True, 1.0, "3", None])
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
    [[], ["open", "close"], ["close", "open", 1], "open", None],
)
def test_bad_column_list_is_reported(columns: object) -> None:
    manifest = _manifest()
    manifest["artifacts"]["bars"]["columns"] = columns
    assert data_manifest_contract_errors(manifest) == ["artifact:bars:columns_invalid"]


def test_every_artifact_is_reported_not_just_the_first() -> None:
    manifest = _manifest()
    manifest["artifacts"]["silver"] = {"sha256": "nope", "rows": -2, "columns": []}
    errors = data_manifest_contract_errors(manifest)
    assert errors == [
        "artifact:silver:sha256_invalid",
        "artifact:silver:rows_invalid",
        "artifact:silver:columns_invalid",
    ]


def test_contract_errors_do_not_mutate_the_manifest() -> None:
    manifest = _manifest()
    before = copy.deepcopy(manifest)
    data_manifest_contract_errors(manifest)
    assert manifest == before


def test_public_provider_requires_a_path_for_file_backed_tapes() -> None:
    for source in ("nasdaq_itch", "fi_2010"):
        cfg = AppConfig.model_validate({"data": {"source": source}})
        with pytest.raises(ValueError, match="requires data.source_path"):
            PublicMarketProvider(cfg)


def test_public_provider_refuses_non_bar_sources() -> None:
    cfg = AppConfig.model_validate({"data": {"source": "fred"}})
    provider = PublicMarketProvider(cfg)
    with pytest.raises(ValueError, match="is not a bar provider"):
        provider.get_bars()
    # Corporate actions and the master are optional for generic sources.
    assert provider.get_corporate_actions().is_empty()
    assert provider.get_security_master().is_empty()


def test_make_provider_refuses_a_registered_non_bar_source() -> None:
    cfg = AppConfig.model_validate({"data": {"source": "fred"}})
    with pytest.raises(ValueError, match="is not a market-bar provider"):
        make_provider(cfg)
