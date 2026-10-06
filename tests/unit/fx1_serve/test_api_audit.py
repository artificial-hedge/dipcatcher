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
    assert blob["claim"]["ok"] is True
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


class _SinkHook:  # minimal handler for _callback_sink tests
    pass


class _BoomSink:
    """Fake sink whose cleanup raises — mirrors a wedged shutdown()."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.server_address = ("127.0.0.1", 0)

    def serve_forever(self) -> None:
        pass

    def shutdown(self) -> None:
        raise RuntimeError("wedged sink")

    def server_close(self) -> None:
        pass


def _boom_ctor(*args: object, **kwargs: object) -> object:
    raise RuntimeError("ctor boom")


def test_callback_sink_restores_env_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent and sentinel caller values both survive a clean sink lifecycle."""
    import os

    from fx1.serve.api_audit import _PRIVATE_NET_ENV, _callback_sink

    monkeypatch.delenv(_PRIVATE_NET_ENV, raising=False)
    with _callback_sink(_SinkHook):
        assert os.environ[_PRIVATE_NET_ENV] == "1"
    assert _PRIVATE_NET_ENV not in os.environ

    monkeypatch.setenv(_PRIVATE_NET_ENV, "sentinel")
    with _callback_sink(_SinkHook):
        assert os.environ[_PRIVATE_NET_ENV] == "1"
    assert os.environ[_PRIVATE_NET_ENV] == "sentinel"


def test_callback_sink_restores_env_on_ctor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing sink constructor must not leak the SSRF opt-in."""
    import http.server
    import os

    from fx1.serve.api_audit import _PRIVATE_NET_ENV, _callback_sink

    monkeypatch.delenv(_PRIVATE_NET_ENV, raising=False)
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", _boom_ctor)
    with pytest.raises(RuntimeError), _callback_sink(_SinkHook):
        pass
    assert _PRIVATE_NET_ENV not in os.environ


def test_callback_sink_restores_env_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing probe inside the sink window must not leak the opt-in."""
    import os

    from fx1.serve.api_audit import _PRIVATE_NET_ENV, _callback_sink

    monkeypatch.setenv(_PRIVATE_NET_ENV, "keep-me")
    with pytest.raises(ValueError), _callback_sink(_SinkHook):
        raise ValueError("probe boom")
    assert os.environ[_PRIVATE_NET_ENV] == "keep-me"


def test_callback_sink_restores_env_on_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising shutdown() must not skip env restoration."""
    import http.server
    import os

    from fx1.serve.api_audit import _PRIVATE_NET_ENV, _callback_sink

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", _BoomSink)
    monkeypatch.delenv(_PRIVATE_NET_ENV, raising=False)
    with pytest.raises(RuntimeError), _callback_sink(_SinkHook):
        pass
    assert _PRIVATE_NET_ENV not in os.environ
