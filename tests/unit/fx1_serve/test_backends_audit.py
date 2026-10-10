"""backends_audit battery tests — every probe holds, the sealed bench
payload verifies, and the battery is deterministic run-to-run."""

from __future__ import annotations

import pytest

from fx1.serve.backends_audit import backends_audit, backends_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_ENV_KEYS = (
    "FX1_API_KEY",
    "MOONSHOT_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_LOCAL_TIMEOUT_S",
    "FX1_LOCAL_START_TIMEOUT_S",
    "FX1_SIGNING_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = backends_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = backends_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = backends_audit_bench()
    b = backends_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.backends_audit.backends_audit", lambda: {})
    blob = backends_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
