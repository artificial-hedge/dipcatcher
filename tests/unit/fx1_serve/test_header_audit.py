"""Header-surface audit evidence and isolated resource lifecycle checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import header_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.header_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert measured
    assert all(value is True for value in measured.values()), measured


def test_header_defect_regressions_stay_pinned(measured: dict[str, bool]) -> None:
    """The four defects this battery found stay fixed: BYOK header
    validation escape to a bare 500, dropped ``X-RateLimit-*`` headers on
    managed-key refusals, the flat error shape on ``Content-Length``
    refusals under ``/v1``, and read-side transport faults escaping the
    backend ``RuntimeError`` convention to a bare 500."""
    assert measured["byok_bad_scheme_refuses_422"] is True
    assert measured["byok_missing_netloc_refuses_422"] is True
    assert measured["byok_overlong_url_refuses_422"] is True
    assert measured["byok_bad_url_anthropic_envelope"] is True
    assert measured["byok_invalid_url_value_redacted"] is True
    assert measured["byok_invalid_key_value_redacted"] is True
    assert measured["rl_headers_on_quota_refusal"] is True
    assert measured["rl_headers_on_scope_refusal"] is True
    assert measured["rl_headers_on_key_window_429"] is True
    assert measured["rl_headers_on_global_429"] is True
    assert measured["envelope_bad_content_length_v1"] is True
    assert measured["envelope_bad_content_length_anthropic"] is True
    assert measured["envelope_bad_content_length_harness"] is True
    assert measured["negative_content_length_refuses"] is True
    assert measured["duplicate_content_length_refuses"] is True
    assert measured["content_length_transfer_encoding_refuses"] is True
    assert measured["missing_length_oversize_refuses_413"] is True
    assert measured["underdeclared_oversize_refuses_413"] is True
    assert measured["timeout_enforced_refusal_502"] is True
    assert measured["timeout_header_enforced"] is True
    assert measured["timeout_harness_enforced"] is True
    assert measured["timeout_stream_cut_is_json_502"] is True
    assert measured["timeout_malformed_payload_502"] is True
    assert measured["timeout_batch_bad_header_fails_lines_closed"] is True


def test_precedence_and_dedup_contracts_stay_pinned(
    measured: dict[str, bool],
) -> None:
    """The measured contracts a caller must be able to rely on: body
    ``fx1.*`` wins over ``X-Fx1-*`` headers, header names bind
    case-insensitively, and ambiguous duplicates refuse before framework
    layers can disagree over first-wins versus last-wins."""
    assert measured["timeout_body_overrides_header"] is True
    assert measured["timeout_header_binds_without_body"] is True
    assert measured["backend_body_overrides_header"] is True
    assert measured["backend_header_binds_without_body"] is True
    assert measured["receipt_hashes_body_overrides_header"] is True
    assert measured["dup_xfx1_refuses_400"] is True
    assert measured["dup_xfx1_reverse_refuses_400"] is True
    assert measured["rid_duplicate_refuses_400"] is True
    assert measured["dup_api_key_refuses_400"] is True
    assert measured["mixed_auth_headers_refuse_400"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "header_audit", lambda: results)
    receipt = audit.header_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "header_audit", lambda: measured)
    receipt = audit.header_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.header_audit_bench()


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
        client, _ = audit._client(audit._SpyResolver())
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
