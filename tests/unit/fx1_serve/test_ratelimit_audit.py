"""Synthetic rate-limit shape/fairness probes and the audit's helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import ratelimit_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_FORMER_DEFECTS = frozenset(
    {
        # the identity map was LRU-bounded only on the refusal path — a
        # spray of distinct hosts that all got *admitted* grew
        # ``_buckets`` without limit; the bound now applies on every
        # admission path.
        "limiter_bound_under_admitted_spray",
        # a sub-1 rps limit admitted one call but declared
        # ``X-RateLimit-Limit: 0`` — the header now reports the real
        # one-token floor.
        "bucket_sub_rps_limit_honest",
        # ingress refusals consumed a global-bucket slot but reported no
        # ``X-RateLimit-*`` trio — the refused caller saw no budget state;
        # they now carry the spent trio like every governed refusal.
        "layered_ingress_refusal_reports_trio",
        # ``fx1_rate_limited_total`` counted terminal ``quota_exceeded``
        # refusals as rate-limit denials — it now counts only real
        # rate-limit refusals (``by_status['429']`` keeps the family).
        "metrics_quota_refusal_not_counted",
    }
)


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.ratelimit_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 93
    assert all(value is True for value in measured.values()), measured


def test_former_defects_now_hold(measured: dict[str, Any]) -> None:
    for name in _FORMER_DEFECTS:
        assert measured.get(name) is True, f"{name} regressed — the former defect is back"


def test_bucket_shape_legs(measured: dict[str, Any]) -> None:
    for name in (
        "bucket_burst_exact_capacity",
        "bucket_remaining_decrements",
        "bucket_trickle_refill_single",
        "bucket_full_refill_restores_capacity",
        "bucket_refused_calls_burn_nothing",
        "bucket_race_refusals_exact",
        "bucket_sub_rps_capacity_floor_one",
        "bucket_fractional_burst_floor",
    ):
        assert measured[name] is True, name


def test_layering_legs(measured: dict[str, Any]) -> None:
    for name in (
        "layered_limiter_precedes_auth",
        "layered_ingress_consumes_before_auth",
        "layered_options_preflight_consumes",
        "layered_key_refusal_burns_global",
        "layered_global_precedes_key",
        "layered_global_429_bills_no_use",
        "layered_hosts_fair",
    ):
        assert measured[name] is True, name


def test_window_composition_legs(measured: dict[str, Any]) -> None:
    for name in (
        "window_scopes_share_pool",
        "window_get_surface_consumes",
        "window_postauth_422_consumes",
        "window_idem_replay_consumes",
        "window_store_false_consumes",
        "window_drain_refused_consumes",
        "window_patch_tighten_refuses",
        "window_self_postconsume_truth",
        "window_keys_isolated",
    ):
        assert measured[name] is True, name


def test_store_and_restart_legs(measured: dict[str, Any]) -> None:
    for name in (
        "store_retry_after_exact",
        "store_refusals_never_reanchor",
        "store_fixed_window_drops_whole",
        "store_quota_precedes_window",
        "store_scope_refusal_consumes_nothing",
        "store_authenticate_race_exact",
        "restart_uses_durable",
        "restart_window_occupancy_fresh",
        "restart_policy_still_enforced",
        "restart_bucket_process_local",
    ):
        assert measured[name] is True, name


@pytest.mark.parametrize(
    "results",
    [{}, {"probe": True}, {"probe": False}, {"probe": 1}, {"probe": None}],
)
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "ratelimit_audit", lambda: results)
    receipt = audit.ratelimit_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("RATELIMIT AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ratelimit_audit", lambda: measured)
    receipt = audit.ratelimit_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.ratelimit_audit_bench()


def test_committed_receipt_matches_fresh_measurement(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    path = Path("receipts/fx1_ratelimit_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema"] == "ratelimit_audit.v1"
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
    monkeypatch.setattr(audit, "ratelimit_audit", lambda: measured)
    fresh = audit.ratelimit_audit_bench()
    for field in (
        "kind",
        "schema",
        "data_label",
        "research_only",
        "live_pnl_claim",
        "claim",
        "coverage",
        "interpretation",
    ):
        assert payload[field] == fresh[field], field


def test_audit_context_restores_environment_and_removes_temp_state_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "operator_state"
    state.mkdir()
    marker = state / "keys.jsonl"
    marker.write_bytes(b"operator data must remain untouched\n")
    before = marker.read_bytes()
    env = {
        "FX1_API_STATE_DIR": str(state),
        "FX1_API_RECEIPTS_DIR": str(state),
        "FX1_FT_DIR": str(state),
        "FX1_API_CORS_ORIGINS": "*",
        "FX1_API_JOB_MAX": "not-a-number",
        "MOONSHOT_API_KEY": "synthetic ambient sentinel",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    temporary: Path | None = None
    with pytest.raises(RuntimeError, match="deliberate failure"), audit._audit_context():
        assert all(name not in os.environ for name in env)
        temporary = audit._temporary_directory()
        (temporary / "test.txt").write_text("synthetic")
        os.environ["FX1_TEMPORARY_AUDIT_VALUE"] = "temporary"
        raise RuntimeError("deliberate failure")
    assert temporary is not None and not temporary.exists()
    assert all(os.environ[name] == value for name, value in env.items())
    assert "FX1_TEMPORARY_AUDIT_VALUE" not in os.environ
    assert marker.read_bytes() == before
