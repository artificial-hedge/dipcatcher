"""Regression checks for Anthropic cursor direction and refused options."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.serve.api import _AnthropicBatchRecord, create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    for name in (
        "FX1_API_KEY",
        "FX1_API_RATE_LIMIT_RPS",
        "FX1_API_STATE_DIR",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def no_backend(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("list/validation requests must not resolve a model backend")

    app = create_app(backend_resolver=no_backend, receipts_dir=tmp_path)
    for index in range(7):
        app.state.abatch_store.put(
            _AnthropicBatchRecord(
                batch_id=f"msgbatch_{index}",
                status="ended",
                created_at=1_700_000_000 + index,
                expires_at=1_700_086_400 + index,
                ended_at=1_700_000_100 + index,
            )
        )
    with TestClient(app) as connection:
        yield connection


def test_after_id_visits_each_batch_once_in_newest_first_order(client: TestClient) -> None:
    params: dict[str, Any] = {"limit": 2}
    ids: list[str] = []
    for _ in range(5):
        response = client.get("/v1/messages/batches", params=params)
        assert response.status_code == 200
        page = response.json()
        ids.extend(item["id"] for item in page["data"])
        if not page["has_more"]:
            break
        params["after_id"] = page["last_id"]
    else:
        pytest.fail("forward pagination did not terminate")
    assert ids == [f"msgbatch_{index}" for index in range(6, -1, -1)]


def test_before_id_returns_nearest_previous_pages(client: TestClient) -> None:
    cursor = "msgbatch_0"
    for expected, has_more in (([2, 1], True), ([4, 3], True), ([6, 5], False)):
        response = client.get("/v1/messages/batches", params={"limit": 2, "before_id": cursor})
        assert response.status_code == 200
        page = response.json()
        assert [row["id"] for row in page["data"]] == [f"msgbatch_{i}" for i in expected]
        assert page["has_more"] is has_more
        cursor = page["first_id"]


@pytest.mark.parametrize("direction", ["before_id", "after_id"])
def test_unknown_cursor_returns_an_empty_page(client: TestClient, direction: str) -> None:
    response = client.get("/v1/messages/batches", params={direction: "msgbatch_missing"})
    assert response.status_code == 200
    assert response.json() == {"data": [], "first_id": None, "last_id": None, "has_more": False}


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_control", {"type": "ephemeral"}),
        ("diagnostics", {"enabled": True}),
        ("output_config", {"effort": "high"}),
    ],
)
@pytest.mark.parametrize("route", ["/v1/messages", "/v1/messages/count_tokens"])
def test_unsupported_options_refuse_before_backend_resolution(
    client: TestClient, field: str, value: dict[str, Any], route: str
) -> None:
    body: dict[str, Any] = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }
    if route == "/v1/messages":
        body["max_tokens"] = 8
    response = client.post(route, json=body)
    assert response.status_code == 422
    result = response.json()
    assert result["type"] == "error"
    assert field in result["error"]["message"]


def test_stock_sdk_auto_paginates_both_directions(client: TestClient) -> None:
    import anthropic
    import httpx2

    async def check() -> None:
        async with (
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=client.app), base_url="http://sdk-cursors"
            ) as http,
            anthropic.AsyncAnthropic(
                base_url="http://sdk-cursors", api_key="synthetic", http_client=http, max_retries=0
            ) as sdk,
        ):
            forward = [batch.id async for batch in sdk.messages.batches.list(limit=2)]
            assert forward == [f"msgbatch_{index}" for index in range(6, -1, -1)]
            backward = [
                batch.id
                async for batch in sdk.messages.batches.list(limit=2, before_id="msgbatch_0")
            ]
            assert backward == [
                "msgbatch_2",
                "msgbatch_1",
                "msgbatch_4",
                "msgbatch_3",
                "msgbatch_6",
                "msgbatch_5",
            ]

    asyncio.run(check())
