"""Contract tests for fx1.forecast.data_audit.

- every probe row is exercised and no verdict is ``fail``
- each of the five audited areas is covered
- the sealed receipt verifies via verify_receipt_payload and its digest
  recomputes over the canonical bytes of the unsealed body
- the written receipt file round-trips and verifies
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.forecast.data_audit import (
    AUDIT_KIND,
    AUDIT_SCHEMA,
    forecast_data_audit,
    forecast_data_audit_bench,
    forecast_data_audit_results,
    write_forecast_data_audit_receipt,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_RECEIPT = REPO_ROOT / "receipts" / "forecast_data_audit.json"

_RESULTS = forecast_data_audit_results()
_PROBE_ROWS = pytest.mark.parametrize(
    "row",
    _RESULTS,
    ids=[f"{r['area']}.{r['probe']}" for r in _RESULTS],
)


@_PROBE_ROWS
def test_every_probe_not_failed(row: dict[str, str]) -> None:
    assert row["verdict"] in {"pass", "fixed", "flag"}, (
        f"{row['area']}.{row['probe']}: {row['detail']}"
    )


def test_probe_rows_are_wellformed() -> None:
    areas = set()
    fixed = []
    flags = []
    for row in _RESULTS:
        assert set(row) == {"probe", "area", "verdict", "detail"}
        assert row["probe"] and row["detail"]
        areas.add(row["area"])
        if row["verdict"] == "fixed":
            fixed.append(row["probe"])
        if row["verdict"] == "flag":
            flags.append(row["probe"])
    assert areas == {
        "schema.feature",
        "schema.forecast",
        "features.pit",
        "features.causality",
        "features.resample",
        "artifacts",
        "determinism",
    }
    # the four defects this lane fixed, pinned by regression probes
    assert "quantile_ordering_partial_null_row" in fixed
    assert "null_availability_rejected_no_cutoff" in fixed
    assert "mixed_revisions_refused" in fixed
    assert "per_bar_release_before_event_refused" in fixed
    assert flags, "surprising-but-tolerated surfaces must stay pinned"


def test_audit_body_contract() -> None:
    body = forecast_data_audit()
    assert body["kind"] == AUDIT_KIND == "forecast_data_audit"
    assert body["schema"] == AUDIT_SCHEMA == "forecast_data_audit.v1"
    assert body["data_label"] == "SYNTHETIC"
    assert body["research_only"] is True
    assert body["live_pnl_claim"] is False
    assert body["claim"]["ok"] is True
    assert body["claim"]["results"] == _RESULTS
    assert body["interpretation"]
    assert "receipt_sha256" not in body


def test_forbidden_metric_keys_absent() -> None:
    forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            found = set()
            for key, value in node.items():
                found.add(str(key))
                found |= keys(value)
            return found
        if isinstance(node, list):
            found = set()
            for value in node:
                found |= keys(value)
            return found
        return set()

    assert not (keys(forecast_data_audit_bench()) & forbidden)


def test_sealed_receipt_verifies() -> None:
    receipt = forecast_data_audit_bench()
    outcome = verify_receipt_payload(json.loads(json.dumps(receipt)))
    assert outcome["valid"], outcome.get("errors")

    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert receipt["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))


def test_seal_is_deterministic() -> None:
    first = forecast_data_audit_bench()
    second = forecast_data_audit_bench()
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_written_receipt_roundtrip_verifies(tmp_path: Path) -> None:
    path = write_forecast_data_audit_receipt(tmp_path)
    assert path.name == "forecast_data_audit.json"
    loaded = json.loads(path.read_text())
    outcome = verify_receipt_payload(loaded, path)
    assert outcome["valid"], outcome.get("errors")


def test_committed_receipt_verifies() -> None:
    if not COMMITTED_RECEIPT.exists():
        pytest.skip("committed receipt not generated yet")
    loaded = json.loads(COMMITTED_RECEIPT.read_text())
    outcome = verify_receipt_payload(loaded, COMMITTED_RECEIPT)
    assert outcome["valid"], outcome.get("errors")
