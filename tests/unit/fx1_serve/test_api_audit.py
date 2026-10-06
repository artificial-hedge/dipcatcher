"""Tests for fx1.serve.api + the api_audit lane."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod
from fx1.harness import Harness
from fx1.serve.api_audit import api_audit, api_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def _client() -> TestClient:
    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    return TestClient(api_mod.create_app(harness=Harness(runner=fake_runner)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = api_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = api_audit_bench()
    falses = [name for name, ok in blob["claim"]["results"].items() if ok is not True]
    assert blob["claim"]["ok"] is True, f"probes failed: {falses}"
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = api_audit_bench()
    b = api_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


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


def test_verify_receipt_route() -> None:
    blob = api_audit_bench()
    resp = _client().post("/receipts/verify", json={"receipt": blob})
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
