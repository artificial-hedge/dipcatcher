"""Synthetic auth-surface probes and the audit's evidence/isolation helpers."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import pytest

from fx1.serve import auth_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.auth_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 182
    assert all(value is True for value in measured.values()), measured


def test_channel_and_ordering_contract(measured: dict[str, Any]) -> None:
    for name in (
        "x_api_key_admits",
        "bearer_admits_v1",
        "bearer_ignored_off_v1",
        "garbage_x_api_key_plus_bearer_refused_400",
        "empty_x_api_key_plus_bearer_refused_400",
        "dup_x_api_key_bad_then_good_refused_400",
        "garbage_admin_route_401_not_403",
        "under_scoped_admin_route_403",
        "expired_admin_route_401_not_403",
        "quota_exhausted_admin_429_not_403",
        "refusal_uniform_harness_body",
        "nonascii_x_api_key_uniform_401",
        "nonascii_bearer_uniform_401",
    ):
        assert measured[name] is True, name


def test_lifecycle_and_metering_contract(measured: dict[str, Any]) -> None:
    for name in (
        "revoked_midflight_admitted_completes",
        "revoked_next_call_401",
        "revoked_admin_cannot_mint",
        "rotate_dead_target_refused",
        "scope_refusal_decrements_nothing",
        "forged_storm_mints_no_record",
        "forged_storm_moves_no_meter",
        "expiry_boundary_exact",
        "rotate_dead_predecessor_mints_dead_successor",
        "dev_mode_bare_loopback_admits",
        "post_provision_bare_401",
        "remote_headers_cannot_spoof_loopback",
        "mint_returns_full_key_once",
        "list_surface_never_serializes_raw",
        "refusal_never_echoes_credential",
        "inflight_write_completes_despite_patch",
        "patched_next_write_403",
    ):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "auth_audit", lambda: results)
    receipt = audit.auth_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("AUTH AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "auth_audit", lambda: measured)
    receipt = audit.auth_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.auth_audit_bench()
    assert "TypeScript client runtime" in receipt["coverage"]["not_executed"]


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


def test_client_uses_private_state_and_closes_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operator = tmp_path / "operator"
    operator.mkdir()
    marker = operator / "keys.jsonl"
    marker.write_text("operator sentinel\n")
    monkeypatch.setenv("FX1_API_STATE_DIR", str(operator))
    monkeypatch.setenv("FX1_FT_DIR", str(operator))
    monkeypatch.setenv("FX1_API_RECEIPTS_DIR", str(operator))
    with audit._audit_context():
        client, _ = audit._client()
        raw, _ = audit._mint(client, {audit._H_KEY: audit._ROOT})
        assert audit._models(client, {audit._H_KEY: raw}).status_code == 200
        executor = cast(Any, client.app).state.jobs_executor
        assert executor.submit(lambda: 3).result(timeout=1) == 3
    assert client.is_closed
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(lambda: None)
    assert list(operator.iterdir()) == [marker]
    assert marker.read_text() == "operator sentinel\n"


def test_refusal_code_reads_every_envelope_grammar() -> None:
    class Resp:
        def __init__(self, body: Any) -> None:
            self._body = body

        def json(self) -> Any:
            return self._body

    assert audit._code(Resp({"detail": "x", "code": "unauthorized"})) == "unauthorized"
    assert audit._code(Resp({"error": {"code": "insufficient_scope"}})) == "insufficient_scope"
    assert audit._code(Resp({"type": "error", "error": {"type": "authentication_error"}})) is None
    assert audit._code(Resp("not-json-dict")) is None


def test_surrogateescaped_environment_key_fails_closed() -> None:
    """Undecodable POSIX environment bytes must not escape as a 500."""
    from fx1.serve.keys import ApiKeyStore

    api_mod = import_module("fx1.serve.api")
    store = ApiKeyStore()
    store.mint()
    response = audit._resolve_auth_unit(
        api_mod,
        audit._request([(b"x-api-key", b"fx1k_forged")]),
        "\udcff",
        store,
    )
    assert audit._status_of(response) == 401
