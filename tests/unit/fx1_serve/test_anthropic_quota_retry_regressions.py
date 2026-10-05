"""Hard budgets stay terminal while elapsed rate-limit windows remain retryable."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.serve.api import create_app


@pytest.fixture
def quota_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FX1_API_KEY", "quota-regression-bootstrap")

    def unexpected_backend(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("key refusal must precede backend resolution")

    with TestClient(create_app(backend_resolver=unexpected_backend)) as client:
        yield client


def _exhaust_one_request(client: TestClient, limits: dict[str, int]) -> dict[str, str]:
    minted = client.post(
        "/harness/keys",
        json=limits,
        headers={"X-API-Key": "quota-regression-bootstrap"},
    )
    assert minted.status_code == 201
    headers = {"X-API-Key": minted.json()["key"]}
    assert client.get("/harness/commands", headers=headers).status_code == 200
    return headers


@pytest.mark.parametrize(
    ("path", "extra_headers"),
    [
        ("/v1/messages", {}),
        ("/v1/models", {"anthropic-version": "2023-06-01"}),
    ],
)
def test_hard_quota_refusal_is_not_retryable(
    quota_client: TestClient, path: str, extra_headers: dict[str, str]
) -> None:
    headers = _exhaust_one_request(quota_client, {"max_requests": 1})
    response = quota_client.request(
        "POST" if path == "/v1/messages" else "GET",
        path,
        headers={**headers, **extra_headers},
    )
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert "retry-after" not in response.headers
    assert response.headers["request-id"] == response.headers["x-request-id"]


def test_rate_limit_window_remains_retryable(quota_client: TestClient) -> None:
    headers = _exhaust_one_request(quota_client, {"rpm": 1})
    response = quota_client.post("/v1/messages", headers=headers)
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "true"
    assert int(response.headers["retry-after"]) >= 1
    assert response.headers["anthropic-ratelimit-requests-limit"] == "1"
