"""Gate for the fx-1 forecast-core adversarial audit lane."""

from __future__ import annotations

import pytest

from fx1.forecast.core_audit import (
    forecast_core_audit,
    forecast_core_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def probes() -> dict[str, object]:
    return forecast_core_audit()


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    return forecast_core_audit_bench()


def test_every_probe_is_bool_or_str(probes: dict[str, object]) -> None:
    assert probes
    for name, value in probes.items():
        assert isinstance(value, (bool, str)), f"{name} -> {type(value).__name__}"
        if isinstance(value, str):
            assert value, f"{name} recorded an empty constant"


def test_every_bool_probe_holds(probes: dict[str, object]) -> None:
    failed = {
        name: value for name, value in probes.items() if isinstance(value, bool) and not value
    }
    assert not failed, f"contract probes failed: {sorted(failed)}"


def test_receipt_seals_and_verifies(receipt: dict[str, object]) -> None:
    result = verify_receipt_payload(receipt, "x.json")
    assert result["valid"], result


def test_receipt_claim_matches_probe_run(receipt: dict[str, object]) -> None:
    claim = receipt["claim"]
    assert isinstance(claim, dict)
    results = claim["results"]
    assert isinstance(results, dict)
    assert results == forecast_core_audit()
    assert claim["ok"] is True


def test_receipt_honesty_envelope(receipt: dict[str, object]) -> None:
    assert receipt["kind"] == "forecast_core_audit"
    assert receipt["schema"] == "forecast_core_audit.v1"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["git_revision"]
    assert receipt["receipt_sha256"]
