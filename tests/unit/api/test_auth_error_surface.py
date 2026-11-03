"""API auth refusals and the error envelopes around a bad request."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse

from quant_fund.api.app import (
    _backtest_artifact_digest,
    _stamp_research_honesty,
    _weights_honesty_envelope,
    api_auth_middleware,
    app,
    get_backtest,
    resolve_allowed_config_path,
)
from quant_fund.api.error_shape import error_shape_audit
from quant_fund.api.middleware_audit import middleware_audit
from quant_fund.api.param_fuzz import fuzz_app
from quant_fund.api.research_api import ResearchApiSettings, create_app
from quant_fund.api.surface_pin import _route_entry, diff


def test_config_path_allowlist_rejects_escape_and_missing_files() -> None:
    with pytest.raises(HTTPException, match="too long") as too_long:
        resolve_allowed_config_path("a" * 513)
    assert too_long.value.status_code == 400
    with pytest.raises(HTTPException, match="configs/") as escaped:
        resolve_allowed_config_path("/etc/passwd")
    assert escaped.value.status_code == 400
    with pytest.raises(HTTPException, match="not found") as missing:
        resolve_allowed_config_path("configs/does-not-exist.yaml")
    assert missing.value.status_code == 404
    resolved = resolve_allowed_config_path("configs/research.yaml")
    assert resolved.name == "research.yaml"
    again = resolve_allowed_config_path(str(resolved))
    assert again == resolved
    bare = resolve_allowed_config_path("research.yaml")
    assert bare == resolved
    stamped = _stamp_research_honesty({"live_pnl_claim": True, "research_only": False})
    assert stamped["live_pnl_claim"] is False
    assert stamped["research_only"] is True
    envelope = _weights_honesty_envelope([{"security_id": "A"}])
    assert envelope["n"] == 1
    assert envelope["live_pnl_claim"] is False
    assert envelope["claim"] == "research_only"


def test_auth_middleware_rejects_bad_length_and_sets_hsts() -> None:
    with TestClient(app, base_url="https://testserver") as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.headers["strict-transport-security"].startswith("max-age=")
        small = client.post(
            "/portfolio/optimize",
            content=b'{"nope": 1}',
            headers={"Content-Type": "application/json"},
        )
        assert small.status_code == 422
        negative = client.post(
            "/portfolio/optimize",
            content=b"{}",
            headers={"Content-Length": "-1", "Content-Type": "application/json"},
        )
        assert negative.status_code == 400
        assert negative.json()["detail"] == "invalid Content-Length"

    import anyio

    async def oversized() -> None:
        body = b"x" * (64 * 1024 + 1)

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/optimize",
            "raw_path": b"/optimize",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }
        request = Request(scope, receive)

        async def call_next(incoming: Request) -> JSONResponse:
            await incoming.body()
            return JSONResponse({"ok": True})

        response = await api_auth_middleware(request, call_next)
        assert response.status_code == 413

    anyio.run(oversized)

    async def ignored_control_frame() -> None:
        async def receive():
            return {"type": "http.disconnect"}

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
        }
        request = Request(scope, receive)

        async def call_next(incoming: Request) -> JSONResponse:
            await incoming.body()
            return JSONResponse({"ok": True})

        with pytest.raises(ClientDisconnect):
            await api_auth_middleware(request, call_next)

    anyio.run(ignored_control_frame)


def test_backtest_lookup_rejects_a_torn_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Data:
        def __init__(self, root: Path) -> None:
            self.root = root

    class _Cfg:
        def __init__(self, root: Path) -> None:
            self.data = _Data(root)

    monkeypatch.setattr(
        sys.modules["quant_fund.api.app"], "_load_cfg", lambda _path: _Cfg(tmp_path)
    )
    with pytest.raises(HTTPException) as bad_id:
        get_backtest("not-hex")
    assert bad_id.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        get_backtest("a" * 16)
    assert missing.value.status_code == 404

    root = tmp_path / "metadata" / "backtests"
    root.mkdir(parents=True)
    artifact_id = "b" * 16
    path = root / f"{artifact_id}.json"
    fills = root / "fills.parquet"
    equity = root / "equity.parquet"
    fills.write_bytes(b"fills")
    equity.write_bytes(b"equity")

    def store(payload: dict[str, object]) -> None:
        path.write_text(json.dumps(payload))

    base = {
        "id": artifact_id,
        "metrics": {},
        "scope": {"rows": 1},
        "fills_path": str(fills),
        "equity_path": str(equity),
        "fills_sha256": "a" * 64,
        "equity_sha256": "b" * 64,
        "claim": "research_only",
        "live_pnl_claim": False,
    }
    store({key: value for key, value in base.items() if key != "claim"})
    with pytest.raises(HTTPException) as incomplete:
        get_backtest(artifact_id)
    assert incomplete.value.status_code == 422

    store({**base, "id": "c" * 16})
    with pytest.raises(HTTPException, match="identity"):
        get_backtest(artifact_id)
    store({**base, "live_pnl_claim": True})
    with pytest.raises(HTTPException, match="live-P&L"):
        get_backtest(artifact_id)
    store({**base, "artifact_sha256": "zz"})
    with pytest.raises(HTTPException, match="invalid digest"):
        get_backtest(artifact_id)
    store({**base, "artifact_sha256": "d" * 64})
    with pytest.raises(HTTPException, match="digest mismatch"):
        get_backtest(artifact_id)
    store({**base, "metrics": {"analytics_export": {"live_pnl_claim": True}}})
    with pytest.raises(HTTPException, match="analytics"):
        get_backtest(artifact_id)
    store({**base, "fills_path": "/etc/passwd"})
    with pytest.raises(HTTPException, match="escapes"):
        get_backtest(artifact_id)
    store({**base, "fills_path": ""})
    with pytest.raises(HTTPException, match="non-empty"):
        get_backtest(artifact_id)
    absent = root / "missing.parquet"
    store({**base, "fills_path": str(absent), "artifact_sha256": None})
    # None digest is dropped by json; write it explicitly as absent by rebuilding.
    payload = {**base, "fills_path": str(absent)}
    payload.pop("artifact_sha256", None)
    store(payload)
    with pytest.raises(HTTPException, match="missing"):
        get_backtest(artifact_id)

    payload = {**base, "fills_sha256": "0" * 64, "equity_sha256": "0" * 64}
    payload["artifact_sha256"] = _backtest_artifact_digest(payload)
    store(payload)
    with pytest.raises(HTTPException, match="hash mismatch"):
        get_backtest(artifact_id)


def test_error_shape_records_a_leak_and_a_missing_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Url:
        def __init__(self, path: str) -> None:
            self.path = path

    class _Request:
        def __init__(self, path: str) -> None:
            self.url = _Url(path)

    class _Response:
        def __init__(self, status: int, text: str, path: str) -> None:
            self.status_code = status
            self.text = text
            self.request = _Request(path)

        def json(self) -> dict[str, object]:
            if self.text.startswith("{"):
                parsed = json.loads(self.text)
                assert isinstance(parsed, dict)
                return parsed
            raise ValueError("not json")

    class _Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def get(self, url: str) -> _Response:
            if url == "/openapi.json":
                spec = {"paths": {"/health": {"get": {}}, "/items/{item_id}": {"get": {}}}}
                return _Response(200, json.dumps(spec), "/openapi.json")
            if "limit=-1" in url or "%00" in url:
                return _Response(500, 'Traceback File "/site-packages"', url)
            return _Response(400, "not-json", url)

        def post(self, url: str) -> _Response:
            return _Response(200, "ok", url)

        def delete(self, url: str) -> _Response:
            return _Response(405, json.dumps({"detail": None}), url)

    monkeypatch.setattr("fastapi.testclient.TestClient", _Client)
    report = error_shape_audit()
    assert report["n_violations"] > 0
    assert report["verdict"] == "violations"
    problems = {item for row in report["violations"] for item in row["problems"]}
    assert "internal_leak" in problems
    assert "no_detail_envelope" in problems
    assert any(item.startswith("status=") for item in problems)


def test_surface_diff_names_a_removed_method_and_a_changed_response() -> None:
    pinned = {
        "/x": {
            "GET": {"params": [], "response_fields": ["a"]},
            "POST": {"params": []},
        }
    }
    live = {"/x": {"GET": {"params": [], "response_fields": ["b"]}}}
    report = diff(pinned, live)
    assert report["breaking"] is True
    assert any(item.startswith("method_removed") for item in report["issues"])
    assert any(item.startswith("response_fields_changed") for item in report["issues"])

    class _Param:
        def __init__(self, kind: object) -> None:
            self.kind = kind
            self.default = None

    def _framework_endpoint() -> None:
        return None

    _framework_endpoint.__module__ = "fastapi.applications"

    class _Route:
        path = "/docs"
        methods = {"GET", "HEAD"}
        endpoint = staticmethod(_framework_endpoint)
        response_model = None

    assert _route_entry(_Route()) is None
    skipped = type("Route", (), {"path": "/x", "methods": set(), "endpoint": None})()
    assert _route_entry(skipped) is None

    def _endpoint(q: int, *args: object, **kwargs: object) -> None:
        del q, args, kwargs

    class _Model:
        model_fields = {"a": object}

    class _Contract:
        path = "/weights"
        methods = {"GET", "HEAD", "OPTIONS"}
        endpoint = staticmethod(_endpoint)
        response_model = _Model()

    entry = _route_entry(_Contract())
    assert entry is not None
    assert "HEAD" not in entry["methods"]
    assert "OPTIONS" not in entry["methods"]
    assert entry["methods"]["GET"]["response_fields"] == ["a"]
    bare = type(
        "Route",
        (),
        {"path": "/bare", "methods": {"GET"}, "endpoint": None, "response_model": None},
    )()
    bare_entry = _route_entry(bare)
    assert bare_entry is not None
    assert bare_entry["methods"]["GET"]["params"] == []
    added = diff({"/weights": {"GET": {"params": []}}}, {"/weights": {"POST": {"params": []}}})
    assert any(item.startswith("method_added") for item in added["issues"])


def test_middleware_restores_a_preexisting_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "already-set")
    report = middleware_audit()
    assert report["key_required"]["right_key_status"] == 200
    assert report["body_limits"]["oversized_declared"] == 413


def test_param_fuzz_counts_a_transport_exception() -> None:
    class _Param:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Dep:
        path_params = [_Param("item_id")]
        query_params = [_Param("limit")]

    class _Route:
        path = "/items/{item_id}"
        methods = {"GET"}
        dependant = _Dep()

    class _App:
        routes = [_Route()]

    class _Client:
        def get(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("transport down")

    report = fuzz_app(_App(), _Client())
    assert report["n_cases"] > 0
    assert report["violations"]
    assert all(item["status"] == 599 for item in report["violations"])

    class _Leaking:
        status_code = 400
        text = "Traceback (most recent call last)"

        def get(self, *args: object, **kwargs: object) -> _Leaking:
            del args, kwargs
            return self

    leaked = fuzz_app(_App(), _Leaking())
    assert any(item["class"] == "traceback_leak" for item in leaked["violations"])


def test_research_api_error_responses_fail_closed(tmp_path: Path) -> None:
    data = tmp_path / "data"
    receipts = tmp_path / "receipts"
    verifier = tmp_path / "verifier"
    artifacts = tmp_path / "artifacts"
    for path in (data, receipts, verifier, artifacts):
        path.mkdir()
    run_id = "a" * 64
    runs = data / "metadata" / "research" / "runs"
    runs.mkdir(parents=True)
    (runs / f"{run_id}.json").write_text(json.dumps({"provenance": []}))
    (data / "metadata" / "research" / "latest.json").write_text("[")
    (receipts / "array.json").write_text("[]")
    (receipts / "noschema.json").write_text(
        json.dumps({"research_only": True, "live_pnl_claim": False})
    )
    result = data / "metadata" / "bench" / "run1"
    result.mkdir(parents=True)
    (result / "manifest.json").write_text("[]")
    (verifier / "notes").mkdir()
    (verifier / "v9").mkdir()
    (artifacts / "broken.json").write_text("{")
    settings = ResearchApiSettings(
        data_root=data,
        receipts_dir=receipts,
        verifier_dir=verifier,
        artifacts_dir=artifacts,
    )
    client = TestClient(create_app(settings))
    listed = client.get("/runs")
    assert listed.status_code == 200
    assert listed.json()["items"][0]["run_id"] == run_id
    latest = client.get("/runs/latest")
    assert latest.status_code == 422
    markdown = client.get(f"/runs/{run_id}/markdown")
    assert markdown.status_code == 404
    verified = client.get("/receipts/noschema/verification")
    assert verified.status_code == 200
    assert verified.json()["valid"] is False
    assert "schema_missing" in verified.json()["errors"]
    unreadable = client.get("/receipts/array")
    assert unreadable.status_code == 422
    results = client.get("/results")
    assert results.status_code == 200
    assert results.json()["items"][0]["has_manifest"] is True
    detail = client.get("/results/bench/run1")
    assert detail.status_code == 200
    assert detail.json()["manifest"] is None
    phase = client.get("/results/bench/run1/verification")
    assert phase.status_code == 200
    assert phase.json()["valid"] is False
    bad_name = client.get("/results/bench/has.dot")
    assert bad_name.status_code == 400
    versions = client.get("/verifier/versions")
    assert versions.status_code == 200
    assert versions.json()["items"] == []
    artifact = client.get("/artifacts/broken")
    assert artifact.status_code == 422

    empty = ResearchApiSettings(
        data_root=tmp_path / "absent-data",
        receipts_dir=tmp_path / "absent-receipts",
        verifier_dir=tmp_path / "absent-verifier",
        artifacts_dir=tmp_path / "absent-artifacts",
    )
    quiet = TestClient(create_app(empty))
    assert quiet.get("/receipts").json()["total"] == 0
    assert quiet.get("/results").json()["total"] == 0
    assert quiet.get("/verifier/versions").json()["items"] == []
    assert quiet.get("/artifacts").json()["items"] == []
    assert quiet.get("/runs/latest").status_code == 404

    many = tmp_path / "many-receipts"
    many.mkdir()
    for index in range(1025):
        (many / f"r{index}.json").write_text("{}")
    crowded = ResearchApiSettings(
        data_root=tmp_path / "absent-data",
        receipts_dir=many,
        verifier_dir=tmp_path / "absent-verifier",
        artifacts_dir=tmp_path / "absent-artifacts",
    )
    listed_receipts = TestClient(create_app(crowded)).get("/receipts")
    assert listed_receipts.status_code == 200
    assert listed_receipts.json()["total"] == 1025
