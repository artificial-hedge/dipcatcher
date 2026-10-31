"""KATs for usage accounting — aggregate_usage + GET /harness/usage + SDK twin."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod
from fx1.sdk import Fx1Harness
from fx1.serve.usage_report import UsageReport, aggregate_usage


class _Backend:
    """Reports usage on every complete."""

    def __init__(self) -> None:
        self._model = "fake-0"
        self.calls = 0

    def complete(self, messages, *, sampling=None):
        self.calls += 1
        self.last_usage = {
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "total_tokens": 10,
            "cached_tokens": 1,
        }
        return "ok"

    def close(self) -> None:
        pass


class _Dead:
    def complete(self, messages, *, sampling=None):
        raise RuntimeError("boom")

    def close(self) -> None:
        pass


def _app() -> TestClient:
    ok = _Backend()
    return TestClient(
        api_mod.create_app(backend_resolver=lambda name, *a, **k: ok if name == "byok" else _Dead())
    )


def _two_ok_one_fail(app: TestClient) -> None:
    for _ in range(2):
        app.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "hi"}]},
        )
    app.post(
        "/harness/complete",
        json={"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_usage_totals_and_splits() -> None:
    app = _app()
    _two_ok_one_fail(app)
    r = app.get("/harness/usage")
    assert r.status_code == 200
    u = r.json()
    t = u["totals"]
    assert (
        t["requests"] == 3
        and t["ok"] == 2
        and t["errors"] == 1
        and t["usage_reported"] == 2
        and t["prompt_tokens"] == 8
        and t["completion_tokens"] == 12
        and t["total_tokens"] == 20
        and t["other_usage"] == {"cached_tokens": 2}
        and t["mean_latency_ms"] is not None
    )
    assert u["records_seen"] == 3 and u["records_dropped"] == 0
    assert u["by_backend"]["byok"]["requests"] == 2
    assert u["by_backend"]["hosted_k3"]["errors"] == 1
    assert u["by_model"]["fake-0"]["requests"] == 2


def test_usage_filters() -> None:
    app = _app()
    _two_ok_one_fail(app)
    assert app.get("/harness/usage?backend=byok").json()["totals"]["requests"] == 2
    assert app.get("/harness/usage?model=fake-0").json()["totals"]["requests"] == 2
    assert app.get("/harness/usage?model=nope").json()["totals"]["requests"] == 0
    assert app.get("/harness/usage?until=1").json()["records_seen"] == 0
    future = 2e9
    assert app.get(f"/harness/usage?since={future}").json()["records_seen"] == 0


def test_usage_bad_window_fails_closed() -> None:
    app = _app()
    assert app.get("/harness/usage?since=2&until=1").status_code == 400
    assert app.get("/harness/usage?since=-1").status_code == 422


def test_usage_empty_log() -> None:
    app = _app()
    u = app.get("/harness/usage").json()
    assert u["totals"]["requests"] == 0 and u["totals"]["mean_latency_ms"] is None
    assert u["by_backend"] == {} and u["by_model"] == {}


def test_ring_eviction_counted() -> None:
    log = api_mod._CompletionLog(cap=2)
    for i in range(4):
        log.append(
            api_mod.CompletionRecord(
                completion_id=f"c{i}",
                backend="byok",
                ok=True,
                latency_ms=1.0,
                at=float(i),
                prompt_sha256="p",
            )
        )
    assert log.dropped == 2 and len(log.all()) == 2
    rep = aggregate_usage(log.all(), cap=log.cap, dropped=log.dropped)
    assert rep.records_seen == 2 and rep.records_dropped == 2 and rep.ring_cap == 2


def test_sdk_usage_twin() -> None:
    ok = _Backend()
    sdk = Fx1Harness(backend_resolver=lambda name, *a, **k: ok if name == "byok" else _Dead())
    sdk.complete(
        [{"role": "user", "content": "hi"}],
        backend="byok",
        byok={"base_url": "http://u.test", "api_key": "k", "model": "m"},
    )
    rep = sdk.usage()
    assert isinstance(rep, UsageReport)
    assert rep.totals.requests == 1 and rep.totals.prompt_tokens == 4
    assert rep.by_model["fake-0"].requests == 1
    assert sdk.usage(model="nope").records_seen == 0


def test_sdk_usage_bad_window_raises() -> None:
    sdk = Fx1Harness(backend_resolver=lambda *a, **k: _Backend())
    with pytest.raises(ValueError):
        sdk.usage(since=2.0, until=1.0)
    with pytest.raises(ValueError):
        sdk.usage(backend="nope")
