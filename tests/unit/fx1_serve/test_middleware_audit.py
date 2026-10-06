"""Middleware/request-lifecycle audit evidence and isolated resource checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import middleware_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.middleware_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert measured
    assert all(value is True for value in measured.values()), measured


def test_unclassified_fault_defect_stays_pinned(measured: dict[str, bool]) -> None:
    """The defect this battery found stays fixed: a fault no route
    classified used to unwind past ``call_next`` to the outermost
    ``ServerErrorMiddleware`` and answer a bare ``text/plain`` 500 — no
    request-id, no security headers, no metrics count, no access-log
    line. The dispatch now envelopes it per dialect and refuses
    input-driven recursion as a validation failure instead."""
    assert measured["unclassified_fault_500_v1"] is True
    assert measured["unclassified_fault_500_harness"] is True
    assert measured["unclassified_fault_500_anthropic"] is True
    assert measured["unclassified_fault_logged_v1"] is True
    assert measured["unclassified_fault_logged_harness"] is True
    assert measured["unclassified_fault_logged_anthropic"] is True
    assert measured["unclassified_fault_counted_v1"] is True
    assert measured["unclassified_fault_counted_harness"] is True
    assert measured["unclassified_fault_counted_anthropic"] is True
    assert measured["unclassified_fault_anthropic_twin"] is True
    assert measured["deep_json_harness_enveloped_422"] is True
    assert measured["deep_json_v1_enveloped_not_bare"] is True
    assert measured["tail_security_headers_enveloped_500"] is True


def test_lifecycle_ordering_contracts_stay_pinned(measured: dict[str, bool]) -> None:
    """The ordering a caller must be able to rely on: the limiter refuses
    before ingress and auth; ingress refuses before auth; auth precedes
    routing (unauthed bogus is 401, never 404) and method handling; the
    drain latch sits behind auth and routing; the request-id echoes on
    every outcome class and lands in the access log."""
    assert measured["ratelimit_before_auth_429"] is True
    assert measured["ratelimit_before_ingress_429"] is True
    assert measured["ingress_before_auth_dup_cl_400"] is True
    assert measured["ingress_before_auth_body_cap_413"] is True
    assert measured["unauthed_bogus_v1_401"] is True
    assert measured["authed_bogus_v1_404"] is True
    assert measured["drain_unauthed_401_first"] is True
    assert measured["drain_authed_bogus_404"] is True
    assert measured["gate_unauthed_never_resolves_401"] is True
    assert measured["rid_echo_on_401"] is True
    assert measured["rid_echo_on_413"] is True
    assert measured["rid_echo_on_422"] is True
    assert measured["rid_in_access_log"] is True
    assert measured["tail_access_log_one_line_per_request"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "middleware_audit", lambda: results)
    receipt = audit.middleware_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "middleware_audit", lambda: measured)
    receipt = audit.middleware_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.middleware_audit_bench()


def test_client_resources_and_ambient_state_survive_probe_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ambient = tmp_path / "operator"
    ambient.mkdir()
    sentinel = ambient / "ft_jobs.jsonl"
    sentinel.write_text("operator data must remain unchanged\n")
    config = {
        "FX1_API_STATE_DIR": str(ambient),
        "FX1_API_RECEIPTS_DIR": str(ambient),
        "FX1_FT_DIR": str(ambient),
        "FX1_SDK_STATE_DIR": str(ambient),
        "FX1_API_JOB_MAX": "invalid ambient value",
        "MOONSHOT_API_KEY": "synthetic ambient sentinel",
        "FX1_API_KEY": "synthetic ambient key",
    }
    for name, value in config.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="deliberate failure"), audit._audit_context():
        assert all(name not in os.environ for name in config)
        temporary = audit._temporary_directory()
        client, _ = audit._client()
        assert client.get("/v1/models").status_code == 200
        executor = client.app.state.jobs_executor
        assert executor.submit(lambda: 2).result(timeout=1) == 2
        raise RuntimeError("deliberate failure")
    assert client.is_closed
    assert not temporary.exists()
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert all(os.environ[name] == value for name, value in config.items())
    assert list(ambient.iterdir()) == [sentinel]
    assert sentinel.read_text() == "operator data must remain unchanged\n"
