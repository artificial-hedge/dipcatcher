"""Tests for fx1.serve.files_audit — the /v1/files lifecycle battery."""

from __future__ import annotations

import pytest

from fx1.serve.files_audit import files_audit, files_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pinned a measured defect which has since been fixed —
# kept as the running ledger of what this battery has caught (mirrors
# the convention in the sibling audit tests).
#  - ``GET /v1/files?order=``/``after``/``before`` refusals escaped the
#    error envelope as unhandled 500s because ``openai_file_list`` did
#    not wrap ``paged_item_list``'s ``OpenAICompatError`` the way every
#    sibling route does; fixed with the same try/except wrap.
#  - ``GET /v1/batches?after=<unknown>`` silently re-served the first
#    page instead of refusing the cursor; now 400 ``invalid_cursor``.
_FORMER_DEFECTS = {
    "list_unknown_after_400_envelope",
    "list_unknown_before_400_envelope",
    "list_bad_order_400_envelope",
    "batches_unknown_after_400_envelope",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = files_audit()
    assert len(results) == 130
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_repaired_defect_probes_hold() -> None:
    """Every former negative probe must keep its corrected contract."""
    results = files_audit()
    assert results.keys() >= _FORMER_DEFECTS
    for name in sorted(_FORMER_DEFECTS):
        assert results[name] is True, f"defect {name} returned"


def test_receipt_verifies() -> None:
    blob = files_audit_bench()
    assert blob["claim"]["ok"] is True
    assert blob["data_label"] == "SYNTHETIC"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = files_audit_bench()
    b = files_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.files_audit as audit_module

    monkeypatch.setattr(audit_module, "files_audit", lambda: {})
    blob = files_audit_bench()
    assert blob["claim"]["results"] == {}
    assert blob["claim"]["ok"] is False
    assert blob["data_label"] == "SYNTHETIC"
