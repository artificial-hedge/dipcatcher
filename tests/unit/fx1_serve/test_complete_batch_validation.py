"""Request-boundary regressions for synchronous completion batches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NoReturn

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fx1.serve.api import CompleteBatchRequest, create_app


def test_complete_batch_rejects_empty_inner_message_list() -> None:
    with pytest.raises(ValidationError) as exc_info:
        CompleteBatchRequest.model_validate(
            {
                "backend": "local_fx1",
                "batch": [[]],
            }
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("batch", 0)
    assert errors[0]["type"] == "too_short"


def test_complete_batch_schema_exposes_inner_minimum() -> None:
    batch_schema = CompleteBatchRequest.model_json_schema()["properties"]["batch"]

    assert batch_schema["minItems"] == 1
    assert batch_schema["maxItems"] == 64
    assert batch_schema["items"]["minItems"] == 1


def test_complete_batch_http_rejects_empty_item_before_backend_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in tuple(os.environ):
        if name.startswith("FX1_API_"):
            monkeypatch.delenv(name)

    def unexpected_backend(name: str, *args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError(f"invalid batch must not resolve backend {name!r}")

    app = create_app(backend_resolver=unexpected_backend, state_dir=tmp_path, rate_limit_rps=0)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/harness/complete/batch",
            json={"backend": "local_fx1", "batch": [[]]},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "too_short",
                "loc": ["body", "batch", 0],
                "msg": "List should have at least 1 item after validation, not 0",
                "input": [],
                "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0},
            }
        ],
        "code": "validation",
    }
