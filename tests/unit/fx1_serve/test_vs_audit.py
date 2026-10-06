"""Tests for fx1.serve.vs_audit — the /v1/vector_stores lifecycle battery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fx1.serve.vs_audit import vs_audit, vs_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pinned a measured defect which has since been fixed —
# kept as the running ledger of what this battery has caught (mirrors
# the convention in the sibling audit tests).
#  - ``POST /v1/vector_stores`` with ``file_ids`` journaled and
#    published the store BEFORE attaching the members, so a bogus or
#    duplicate member 404/409ed while leaving a partial store listed;
#    create now rolls back through the delete tombstone.
#  - vector-store create/update/attach/file-batch ran under the drain
#    latch — the only mutation surface without the submit gate; all
#    four now answer ``503 draining`` after the idempotency replay
#    check.
#  - the three vector-store mints ignored ``Idempotency-Key`` entirely
#    — a retry minted a second store or 409ed on its own attachment;
#    all now dedupe under ``upload_idem_store`` with per-route and
#    per-store namespaces.
#  - the three vector-store list routes declared
#    ``Literal["asc", "desc"]`` so a bad ``order`` answered a bare
#    pydantic 422 instead of the shared fail-closed ``400
#    invalid_cursor`` contract the sibling surfaces pin.
_FORMER_DEFECTS = {
    "create_file_ids_bogus_no_partial_store",
    "create_file_ids_mixed_404_no_partial_store",
    "create_file_ids_dup_no_partial_store",
    "create_rollback_frees_member_files",
    "drain_create_503",
    "drain_update_503",
    "drain_attach_503",
    "drain_batch_503",
    "drain_idem_replay_bypasses_gate",
    "idem_create_replays_same_id",
    "idem_create_replay_header",
    "idem_create_replay_byte_identical",
    "idem_create_conflict_409",
    "idem_attach_replays_not_409",
    "idem_attach_conflict_409",
    "idem_batch_replays_same_batch",
    "idem_batch_conflict_409",
    "idem_key_namespaced_by_route_and_store",
    "files_list_bad_order_400",
    "list_bad_order_400",
    "batch_files_bad_order_400",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return vs_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 277
    for name, ok in measured.items():
        assert ok is True, f"probe {name} failed"


def test_repaired_defect_probes_hold(measured: dict[str, bool]) -> None:
    """Every former negative probe must keep its corrected contract."""
    assert measured.keys() >= _FORMER_DEFECTS
    for name in sorted(_FORMER_DEFECTS):
        assert measured[name] is True, f"defect {name} returned"


def test_receipt_verifies(measured: dict[str, bool]) -> None:
    blob = vs_audit_bench(measured)
    assert blob["claim"]["ok"] is True
    assert blob["data_label"] == "SYNTHETIC"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic(measured: dict[str, bool]) -> None:
    a = vs_audit_bench(measured)
    b = vs_audit_bench(measured)
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_receipt_rejects_incomplete_or_non_boolean_measurements() -> None:
    revision = "0" * 40
    empty = vs_audit_bench({}, revision=revision)
    false_result = vs_audit_bench({f"probe_{i}": i != 1 for i in range(277)}, revision=revision)
    truthy_int = vs_audit_bench(
        cast(dict[str, bool], {f"probe_{i}": 1 if i == 1 else True for i in range(277)}),
        revision=revision,
    )
    assert empty["claim"]["ok"] is False
    assert false_result["claim"]["ok"] is False
    assert truthy_int["claim"]["ok"] is False


def test_committed_receipt_matches_fresh_measurement(measured: dict[str, bool]) -> None:
    committed: dict[str, Any] = json.loads(Path("receipts/fx1_vs_audit.json").read_text())
    assert verify_receipt_payload(committed)["valid"] is True
    fresh = vs_audit_bench(measured, revision=str(committed["git_revision"]))
    assert committed == fresh


def test_empty_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.vs_audit as audit_module

    monkeypatch.setattr(audit_module, "vs_audit", lambda: {})
    blob = vs_audit_bench()
    assert blob["claim"]["results"] == {}
    assert blob["claim"]["ok"] is False
    assert blob["data_label"] == "SYNTHETIC"


def test_audit_shuts_down_every_created_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit-owned apps must not strand their job-worker pools."""
    import fx1.serve.api as api_module

    real_executor = api_module.ThreadPoolExecutor
    created: list[object] = []

    class TrackingExecutor(real_executor):
        shutdown_called = False

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

        def shutdown(self, *args: object, **kwargs: object) -> None:
            self.shutdown_called = True
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(api_module, "ThreadPoolExecutor", TrackingExecutor)
    vs_audit()
    assert created
    assert all(getattr(executor, "shutdown_called", False) for executor in created)
