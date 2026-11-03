"""Tests for fx1.serve.api + the api_audit lane."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod
from fx1.harness import Harness
from fx1.serve.api_audit import api_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload
from tests.unit.fx1_serve.audit_session import run_with_clean_env

_AUDIT_ENV = ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR")


def _client() -> TestClient:
    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    return TestClient(api_mod.create_app(harness=Harness(runner=fake_runner)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AUDIT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def api_bench() -> dict[str, Any]:
    return run_with_clean_env(api_audit_bench, _AUDIT_ENV)


def test_all_probes_hold(api_bench: dict[str, Any]) -> None:
    results = api_bench["claim"]["results"]
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies(api_bench: dict[str, Any]) -> None:
    assert api_bench["claim"]["ok"] is True
    verdict = verify_receipt_payload(api_bench)
    assert verdict["valid"] is True


def test_receipt_deterministic(api_bench: dict[str, Any]) -> None:
    second = api_audit_bench()
    assert api_bench["receipt_sha256"] == second["receipt_sha256"]


def test_commands_match_registry() -> None:
    resp = _client().get("/harness/commands")
    assert resp.status_code == 200
    from fx1.harness import HARNESS_REGISTRY

    assert {c["name"] for c in resp.json()["items"]} == {c.name for c in HARNESS_REGISTRY}


def test_unknown_command_fails_closed() -> None:
    resp = _client().post("/harness/runs", json={"command": "not-a-command"})
    assert resp.status_code == 404


def test_config_containment_over_http() -> None:
    resp = _client().post(
        "/harness/runs",
        json={"command": "doctor", "config": "/etc/passwd"},
    )
    assert resp.status_code == 422


def test_verify_receipt_route(api_bench: dict[str, Any]) -> None:
    resp = _client().post("/receipts/verify", json={"receipt": api_bench})
    assert resp.status_code == 200
    assert resp.json()["valid"] is True


def test_verify_receipt_route_fails_closed_on_verifier_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verifier_error(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("forced verifier exception")

    monkeypatch.setattr(api_mod, "verify_receipt_payload", verifier_error)
    resp = _client().post("/receipts/verify", json={"receipt": {}})

    assert resp.status_code == 200
    assert resp.json() == {
        "valid": False,
        "path": "<api>",
        "schema_tag": "",
        "kind": None,
        "verdict": None,
        "digest_convention": None,
        "errors": ["forced verifier exception"],
        "warnings": [],
    }
