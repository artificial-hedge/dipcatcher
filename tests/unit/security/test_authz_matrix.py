"""Every lab and research route, against four credentials.

A route that appears on the app and is missing from the table fails the test.
The table lists the methods FastAPI registers. Decorated GET routes do not
register HEAD; ``/openapi.json`` does.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from quant_fund.api.research_api import ResearchApiSettings, create_app
from quant_fund.schemas.errors import OptimizationInfeasible

_ADMIN = "matrix-admin-key"
_READ = "matrix-read-key"
_WRONG = "matrix-wrong-key"
_KEYS = (_ADMIN, _READ, _WRONG)
_RUN = "ab" * 32
_DIGEST = "cd" * 32
# none, wrong, read, admin
_Status = tuple[int, int, int, int]

_LAB: dict[tuple[str, str], _Status] = {
    ("GET", "/health"): (200, 200, 200, 200),
    ("GET", "/openapi.json"): (401, 401, 200, 200),
    ("HEAD", "/openapi.json"): (401, 401, 200, 200),
    ("GET", "/ready"): (401, 401, 503, 503),
    ("GET", "/models"): (401, 401, 200, 200),
    ("GET", "/forecast/{symbol}"): (401, 401, 200, 200),
    ("GET", "/forecast/{symbol}/distribution"): (401, 401, 200, 200),
    ("GET", "/ranking"): (401, 401, 200, 200),
    ("GET", "/regime"): (401, 401, 200, 200),
    ("GET", "/risk/portfolio"): (401, 401, 200, 200),
    ("GET", "/portfolio/target"): (401, 401, 200, 200),
    ("GET", "/portfolio/exposures"): (401, 401, 200, 200),
    ("GET", "/backtest/{backtest_id}"): (401, 401, 404, 404),
    ("GET", "/monitoring/drift"): (401, 401, 200, 200),
    ("GET", "/doctor"): (401, 401, 200, 200),
    ("GET", "/research/latest"): (401, 401, 404, 404),
    ("POST", "/portfolio/optimize"): (401, 401, 403, 200),
    ("POST", "/backtest"): (401, 401, 403, 422),
}
_LAB_CONFIG = frozenset(
    {
        "/ready",
        "/forecast/{symbol}",
        "/forecast/{symbol}/distribution",
        "/ranking",
        "/risk/portfolio",
        "/portfolio/target",
        "/portfolio/exposures",
        "/backtest/{backtest_id}",
        "/monitoring/drift",
        "/doctor",
        "/research/latest",
    }
)

_RESEARCH: dict[tuple[str, str], _Status] = {
    ("GET", "/health"): (200, 200, 200, 200),
    ("GET", "/ready"): (200, 200, 200, 200),
    ("GET", "/openapi.json"): (401, 401, 200, 200),
    ("HEAD", "/openapi.json"): (401, 401, 200, 200),
    ("GET", "/runs"): (401, 401, 200, 200),
    ("GET", "/runs/latest"): (401, 401, 404, 404),
    ("GET", "/runs/{run_id}"): (401, 401, 404, 404),
    ("GET", "/runs/{run_id}/markdown"): (401, 401, 404, 404),
    ("GET", "/runs/{run_id}/verification"): (401, 401, 404, 404),
    ("GET", "/receipts"): (401, 401, 200, 200),
    ("GET", "/receipts/by-hash/{digest}"): (401, 401, 404, 404),
    ("GET", "/receipts/{receipt_id}"): (401, 401, 404, 404),
    ("GET", "/receipts/{receipt_id}/verification"): (401, 401, 404, 404),
    ("GET", "/results"): (401, 401, 200, 200),
    ("GET", "/results/{kind}/{name}"): (401, 401, 404, 404),
    ("GET", "/results/{kind}/{name}/verification"): (401, 401, 404, 404),
    ("GET", "/verifier/versions"): (401, 401, 200, 200),
    ("GET", "/verifier/versions/{version}"): (401, 401, 404, 404),
    ("GET", "/verifier/runs"): (401, 401, 200, 200),
    ("GET", "/verifier/runs/{run_log_id}"): (401, 401, 404, 404),
    ("GET", "/artifacts"): (401, 401, 200, 200),
    ("GET", "/artifacts/{artifact_id:path}"): (401, 401, 404, 404),
}


class _Point:
    symbol = "AAPL"
    security_id = "AAPL"
    rank_percentile = {"5d": 0.0}

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        del mode
        return {
            "symbol": "AAPL",
            "quantiles": {"0.5": 0.0},
            "interval_lo": 0.0,
            "interval_hi": 0.0,
            "interval_alpha": 0.1,
            "interval_method": "synthetic",
        }


class _State:
    forecasts = [_Point()]


def _forecast(_cfg: object) -> _State:
    return _State()


def _weights(_cfg: object, _asof: object = None) -> pl.DataFrame:
    del _cfg, _asof
    return pl.DataFrame({"target_weight": [1.0]})


def _silver(_cfg: object) -> pl.DataFrame:
    del _cfg
    from datetime import UTC, datetime

    return pl.DataFrame(
        {
            "event_time": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
            ]
        }
    )


def _dates(_cfg: object, dates: list[object]) -> list[object]:
    del _cfg
    return list(dates)


def _infeasible(*_args: object, **_kwargs: object) -> None:
    raise OptimizationInfeasible("interval caps")


def _routes(application: FastAPI) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for route in application.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not isinstance(path, str):
            continue
        for method in methods:
            found.add((str(method), path))
    return found


def _concrete(template: str) -> str:
    return (
        template.replace("{symbol}", "AAPL")
        .replace("{backtest_id}", "0123456789abcdef")
        .replace("{run_id}", _RUN)
        .replace("{digest}", _DIGEST)
        .replace("{receipt_id}", "receipt1")
        .replace("{kind}", "benchmark")
        .replace("{name}", "sample")
        .replace("{version}", "v1")
        .replace("{run_log_id}", "20200101T000000Z")
        .replace("{artifact_id:path}", "artifact1")
    )


def _assert_covered(application: FastAPI, rows: Mapping[tuple[str, str], _Status]) -> None:
    found = _routes(application)
    expected = set(rows)
    missing = sorted(found - expected)
    stale = sorted(expected - found)
    assert not missing and not stale, f"uncovered={missing} stale={stale}"


def _headers(token: str | None) -> dict[str, str]:
    if token is None:
        return {}
    return {"X-API-Key": token}


def _send(
    client: TestClient,
    method: str,
    template: str,
    token: str | None,
    *,
    config: bool,
) -> Response:
    headers = _headers(token)
    path = _concrete(template)
    if method == "POST":
        return client.post(path, headers=headers, json={"config_path": "research.yaml"})
    params = {"config_path": "research.yaml"} if config else None
    if method == "HEAD":
        return client.head(path, headers=headers, params=params)
    return client.request(method, path, headers=headers, params=params)


def _check(response: Response, status: int, method: str) -> None:
    assert response.status_code == status, response.text
    for key in _KEYS:
        assert key not in response.text
    if method == "HEAD":
        return
    if status == 401:
        assert response.json()["detail"] == "invalid or missing X-API-Key"
    if status == 403:
        assert response.json() == {"detail": "read key cannot modify"}


def _exercise(
    client: TestClient,
    rows: Mapping[tuple[str, str], _Status],
    *,
    config_paths: frozenset[str],
) -> None:
    order = ("none", "wrong", "read", "admin")
    tokens = (None, _WRONG, _READ, _ADMIN)
    for (method, template), expected in rows.items():
        for name, token, status in zip(order, tokens, expected, strict=True):
            response = _send(
                client,
                method,
                template,
                token,
                config=template in config_paths,
            )
            try:
                _check(response, status, method)
            except AssertionError as exc:
                raise AssertionError(f"{method} {template} {name}: {exc}") from exc


@pytest.fixture
def lab_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("QUANT_API_KEY", _ADMIN)
    monkeypatch.setenv("QUANT_API_READ_KEY", _READ)
    api = importlib.import_module("quant_fund.api.app")
    root = tmp_path.resolve()
    (root / "research.yaml").write_text(
        f"runtime:\n  mode: research\ndata:\n  root: {root.as_posix()}\n  source: synthetic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "_CONFIGS_DIR", root)
    monkeypatch.setattr(api, "forecast_asof", _forecast)
    monkeypatch.setattr(api, "optimize_asof", _weights)
    monkeypatch.setattr(api, "_causal_decision_dates", _dates)
    monkeypatch.setattr(api, "build_causal_weight_panel", _infeasible)
    monkeypatch.setattr("quant_fund.pipeline.dataset.ensure_silver", _silver)
    return TestClient(api.app, raise_server_exceptions=False)


@pytest.fixture
def research_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("RESEARCH_API_READ_KEY", _READ)
    settings = ResearchApiSettings(
        data_root=tmp_path / "data",
        receipts_dir=tmp_path / "receipts",
        verifier_dir=tmp_path / "verifier",
        artifacts_dir=tmp_path / "artifacts",
        api_key=_ADMIN,
    )
    for path in (
        settings.data_root,
        settings.receipts_dir,
        settings.verifier_dir,
        settings.artifacts_dir,
    ):
        path.mkdir()
    return TestClient(create_app(settings), raise_server_exceptions=False)


def test_lab_matrix_covers_every_route(lab_client: TestClient) -> None:
    app = lab_client.app
    assert isinstance(app, FastAPI)
    _assert_covered(app, _LAB)
    _exercise(lab_client, _LAB, config_paths=_LAB_CONFIG)


def test_research_matrix_covers_every_route(research_client: TestClient) -> None:
    app = research_client.app
    assert isinstance(app, FastAPI)
    _assert_covered(app, _RESEARCH)
    _exercise(research_client, _RESEARCH, config_paths=frozenset())
