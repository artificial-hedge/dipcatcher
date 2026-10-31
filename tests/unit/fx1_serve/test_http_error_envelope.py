"""Routing refusals and endpoint errors share the public error contract."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, NoReturn

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.serve.api import ApiError, create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    for name in tuple(os.environ):
        if name.startswith("FX1_API_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FX1_API_KEY", "routing-regression-key")

    def unused_runner(argv: list[str], timeout_s: int) -> NoReturn:
        raise AssertionError("routing tests must not execute a harness command")

    def unused_backend(name: str, *args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("routing tests must not resolve a backend")

    app = create_app(
        harness=Harness(runner=unused_runner),
        backend_resolver=unused_backend,
        rate_limit_rps=0,
        cors_origins="",
    )

    @app.get("/harness/test-http-error", response_model=None)
    def http_error() -> NoReturn:
        raise HTTPException(429, "retry later", headers={"Retry-After": "7"})

    @app.get("/harness/test-api-error", response_model=None)
    def api_error() -> NoReturn:
        raise ApiError(503, "draining", code="draining", headers={"Retry-After": "9"})

    test_client = TestClient(
        app,
        headers={"X-API-Key": "routing-regression-key", "X-Request-ID": "routing-regression"},
    )
    try:
        with test_client:
            yield test_client
    finally:
        test_client.close()
        app.state.jobs_executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize(
    ("method", "path", "status", "body", "headers"),
    [
        (
            "GET",
            "/harness/no-such-route",
            404,
            {"detail": "Not Found", "code": "not_found"},
            {},
        ),
        (
            "POST",
            "/health",
            405,
            {"detail": "Method Not Allowed", "code": "method_not_allowed"},
            {"allow": "GET"},
        ),
        (
            "GET",
            "/harness/test-http-error",
            429,
            {"detail": "retry later", "code": "too_many_requests"},
            {"retry-after": "7"},
        ),
        (
            "GET",
            "/harness/test-api-error",
            503,
            {"detail": "draining", "code": "draining"},
            {"retry-after": "9"},
        ),
    ],
)
def test_http_error_envelope(
    client: TestClient,
    method: str,
    path: str,
    status: int,
    body: dict[str, str],
    headers: dict[str, str],
) -> None:
    response = client.request(method, path)
    assert response.status_code == status
    assert response.json() == body
    assert response.headers["x-request-id"] == "routing-regression"
    for name, value in headers.items():
        assert response.headers[name] == value


def test_head_routing_refusal_preserves_allow(client: TestClient) -> None:
    response = client.head("/health")
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.content == b""


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/v1/no-such-route"), ("POST", "/v1/models")],
)
def test_openai_catch_all_keeps_provider_envelope(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": f"Invalid URL ({method} {path})",
            "type": "invalid_request_error",
            "param": None,
            "code": "not_found",
        }
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/v1/messages/no-such-route"), ("GET", "/v1/messages")],
)
def test_anthropic_catch_all_keeps_provider_envelope(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path)
    assert response.status_code == 404
    assert response.json() == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "Not Found"},
    }
