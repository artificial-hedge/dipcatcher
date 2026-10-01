"""Tests for api/param_fuzz.py."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_fund.api.param_fuzz import api_fuzz_bench, fuzz_app


def _app(ok: bool = True) -> FastAPI:
    app = FastAPI()

    @app.get("/probe/{pid}")
    def probe(pid: str, limit: int = 5, mode: str = "x"):
        return {"pid": pid, "limit": limit, "mode": mode}

    @app.get("/ok")
    def ok_():
        return {"ok": True}

    return app


def test_clean_app_reports_no_violations():
    app = _app()
    client = TestClient(app)
    out = fuzz_app(app, client)
    assert out["routes_fuzzed"] == 2
    assert out["clean"], out["violations"]


def test_500_is_a_violation():
    app = _app()

    @app.get("/crash/{cid}")
    def crash(cid: str):
        if len(cid) > 1000:
            raise RuntimeError("boom")
        return {}

    client = TestClient(app, raise_server_exceptions=False)
    out = fuzz_app(app, client)
    assert not out["clean"]
    assert any(
        v["class"] == "server_error" and v["route"] == "/crash/{cid}" for v in out["violations"]
    )


def test_counts_and_determinism():
    app = _app()
    a = fuzz_app(app, TestClient(app))
    b = fuzz_app(app, TestClient(app))
    assert a["n_cases"] == b["n_cases"] and a["n_cases"] > 0
    assert a["per_route_cases"] == b["per_route_cases"]


def test_bench_sealed():
    r1 = api_fuzz_bench(_app(), TestClient(_app()))
    r2 = api_fuzz_bench(_app(), TestClient(_app()))
    assert r1["schema"] == "api_fuzz.v1"
    assert r1["claim"] == "no_5xx_on_adversarial_params"
    # sealed payload must be byte-reproducible across runs
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert r1 == r2


@pytest.mark.parametrize("edge", ["../etc/passwd", "%2e%2e%2fetc", "' OR 1=1--"])
def test_traversal_and_sqli_rejected_cleanly(edge):
    from urllib.parse import quote

    app = _app()
    client = TestClient(app)
    resp = client.get("/probe/" + quote(edge, safe=""))
    assert resp.status_code < 500
