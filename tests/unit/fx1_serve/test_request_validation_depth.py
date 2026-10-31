"""Validation errors must remain client errors when rejected inputs are deep."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from fx1.harness import Harness
from fx1.serve.api import create_app


@pytest.fixture
def app_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[FastAPI, TestClient, list[list[str]]]]:
    for name in tuple(os.environ):
        if name.startswith("FX1_"):
            monkeypatch.delenv(name)
    calls: list[list[str]] = []

    def runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        calls.append(argv)
        return 0, "ok", ""

    app = create_app(harness=Harness(runner=runner), state_dir=tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield app, client, calls


@pytest.mark.parametrize("path", ["/harness/runs", "/v1/chat/completions", "/v1/messages"])
def test_deep_validation_error_preserves_each_dialect(
    app_client: tuple[FastAPI, TestClient, list[list[str]]], path: str
) -> None:
    app, _, calls = app_client
    nested: Any = []
    for _ in range(3000):
        nested = [nested]
    message = "Extra inputs are not permitted"
    exc = RequestValidationError(
        [
            {
                "type": "extra_forbidden",
                "loc": ("body", "x"),
                "msg": message,
                "input": nested,
            }
        ]
    )
    # Exercise the actual registered handler without depending on a JSON
    # decoder's separate, Python-version-specific nesting limit.
    request = Request({"type": "http", "path": path, "headers": []})
    response = asyncio.run(app.exception_handlers[RequestValidationError](request, exc))
    assert response.status_code == 422
    payload = json.loads(response.body)
    if path == "/harness/runs":
        assert payload == {
            "detail": [{"type": "extra_forbidden", "loc": ["body", "x"], "msg": message}],
            "code": "validation",
        }
    else:
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["message"] == f"body.x: {message}"
        if path == "/v1/messages":
            assert payload["type"] == "error"
        else:
            assert payload["error"]["code"] == "validation"
    assert calls == []


def test_bounded_deep_json_request_is_a_structured_client_error(
    app_client: tuple[FastAPI, TestClient, list[list[str]]],
) -> None:
    _, client, calls = app_client
    body = b'{"command":"doctor","x":' + b"[" * 3000 + b"]" * 3000 + b"}"
    assert len(body) == 6025
    response = client.post(
        "/harness/runs",
        content=body,
        headers={"content-type": "application/json", "x-request-id": "depth-regression"},
    )
    assert response.status_code in (400, 422)
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == "depth-regression"
    payload = response.json()
    assert "detail" in payload
    if response.status_code == 422:
        assert payload["code"] == "validation"
        assert payload["detail"][0]["loc"] == ["body", "x"]
        assert "input" not in payload["detail"][0]
    assert calls == []


def test_shallow_validation_error_keeps_existing_input_detail(
    app_client: tuple[FastAPI, TestClient, list[list[str]]],
) -> None:
    _, client, calls = app_client
    rejected = {"nested": [1, 2]}
    response = client.post("/harness/runs", json={"command": "doctor", "x": rejected})
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "extra_forbidden",
                "loc": ["body", "x"],
                "msg": "Extra inputs are not permitted",
                "input": rejected,
            }
        ],
        "code": "validation",
    }
    assert calls == []


def test_invalid_json_keeps_existing_error_envelope(
    app_client: tuple[FastAPI, TestClient, list[list[str]]],
) -> None:
    _, client, calls = app_client
    response = client.post(
        "/harness/runs", content=b'{"command":', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation"
    assert response.json()["detail"][0]["type"] == "json_invalid"
    assert calls == []


def test_valid_request_still_executes_once(
    app_client: tuple[FastAPI, TestClient, list[list[str]]],
) -> None:
    _, client, calls = app_client
    response = client.post("/harness/runs", json={"command": "doctor"})
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert len(calls) == 1
