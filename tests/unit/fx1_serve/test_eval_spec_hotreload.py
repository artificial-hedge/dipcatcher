"""Eval-spec hot-reload — declared shape editable while unbound, frozen
once a run binds, on the wire, the SDK twin, and the HTTP client twin."""

from __future__ import annotations

import json
import urllib.parse
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.client import HarnessClient, HarnessTransportError

_SCHEMA = {"suite": "tooluse", "seed": 11, "backend": "byok"}
_SPEC_BODY = {
    "name": "reload-bank",
    "data_source_config": {"type": "custom", "item_schema": _SCHEMA},
    "testing_criteria": [{"name": "all-pass"}],
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FX1_API_KEY",
        "FX1_BYOK_BASE_URL",
        "FX1_BYOK_API_KEY",
        "FX1_BYOK_MODEL",
        "FX1_LOCAL_SERVE_URL",
        "FX1_LOCAL_SERVE_CMD",
        "FX1_LOCAL_MODEL",
        "FX1_LOCAL_API_KEY",
        "FX1_API_STATE_DIR",
        "FX1_SDK_STATE_DIR",
        "FX1_CHECKPOINT_DIR",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _spec_id(client: TestClient) -> str:
    spec = client.post("/v1/evals", json=_SPEC_BODY)
    assert spec.status_code == 201, spec.text
    return str(spec.json()["id"])


def test_wire_unbound_shape_reloads() -> None:
    client = TestClient(create_app())
    spec_id = _spec_id(client)
    res = client.post(
        f"/v1/evals/{spec_id}",
        json={
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 13, "backend": "byok"},
            },
            "testing_criteria": [{"name": "c1"}, {"name": "c2"}],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["data_source_config"]["item_schema"]["seed"] == 13
    assert [c["name"] for c in body["testing_criteria"]] == ["c1", "c2"]
    got = client.get(f"/v1/evals/{spec_id}").json()
    assert got["data_source_config"]["item_schema"]["seed"] == 13


def test_wire_bound_shape_frozen_409() -> None:
    client = TestClient(create_app())
    spec_id = _spec_id(client)
    run = client.post(f"/v1/evals/{spec_id}/runs", json={"model": "byok"})
    assert run.status_code == 201, run.text
    for body in (
        {"data_source_config": {"type": "custom", "item_schema": _SCHEMA}},
        {"testing_criteria": [{"name": "c1"}]},
    ):
        res = client.post(f"/v1/evals/{spec_id}", json=body)
        assert res.status_code == 409, res.text
        assert res.json()["error"]["code"] == "eval_spec_frozen"
    meta = client.post(f"/v1/evals/{spec_id}", json={"name": "still-editable"})
    assert meta.status_code == 200
    assert meta.json()["name"] == "still-editable"


def test_wire_shape_validation_still_422() -> None:
    client = TestClient(create_app())
    spec_id = _spec_id(client)
    bad = client.post(
        f"/v1/evals/{spec_id}",
        json={"data_source_config": {"type": "custom", "item_schema": {}}},
    )
    assert bad.status_code == 422
    none = client.post(f"/v1/evals/{spec_id}", json={})
    assert none.status_code == 422


def test_sdk_twin_bound_frozen() -> None:
    harness = Fx1Harness()
    spec = harness.eval_spec_create(
        "reload-bank",
        suite="tooluse",
        seed=11,
        backend="byok",
    )
    spec_id = str(spec["id"])
    reloaded = harness.eval_spec_update(
        spec_id,
        data_source_config={
            "type": "custom",
            "item_schema": {"suite": "tooluse", "seed": 13, "backend": "byok"},
        },
    )
    assert reloaded["data_source_config"]["item_schema"]["seed"] == 13
    harness.eval_run_create(spec_id, model="byok")
    with pytest.raises(RuntimeError, match="frozen"):
        harness.eval_spec_update(
            spec_id,
            data_source_config={"type": "custom", "item_schema": _SCHEMA},
        )
    renamed = harness.eval_spec_update(spec_id, name="still-editable")
    assert renamed["name"] == "still-editable"


def _transport_for(client: TestClient):
    def _t(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, dict[str, str], bytes]:
        path = urllib.parse.urlsplit(url).path
        res = client.request(method, path, json=payload, headers=headers or None)
        return res.status_code, dict(res.headers), res.content

    return _t


def test_client_twin_forwards_shape_fields() -> None:
    tc = TestClient(create_app())
    api = HarnessClient("http://testclient", api_key="k3y-material", transport=_transport_for(tc))
    spec_id = _spec_id(tc)
    body = api.eval_spec_update(
        spec_id,
        data_source_config={
            "type": "custom",
            "item_schema": {"suite": "tooluse", "seed": 21, "backend": "byok"},
        },
        testing_criteria=[{"name": "c1"}],
    )
    assert body["data_source_config"]["item_schema"]["seed"] == 21
    assert body["testing_criteria"][0]["name"] == "c1"
    api.eval_run_create(spec_id, model="byok")
    with pytest.raises(HarnessTransportError) as err:
        api.eval_spec_update(spec_id, data_source_config={"type": "custom", "item_schema": _SCHEMA})
    assert err.value.code == "eval_spec_frozen"


def test_bound_freeze_survives_state_dir_replay() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "state"
        app = create_app(state_dir=state)
        client = TestClient(app)
        spec_id = _spec_id(client)
        assert client.post(f"/v1/evals/{spec_id}/runs", json={"model": "byok"}).status_code == 201
        # A second app over the same journal replays the binding — the
        # freeze is evidence, not an in-memory flag.
        client2 = TestClient(create_app(state_dir=state))
        res = client2.post(
            f"/v1/evals/{spec_id}",
            json={"data_source_config": {"type": "custom", "item_schema": _SCHEMA}},
        )
        assert res.status_code == 409
        assert res.json()["error"]["code"] == "eval_spec_frozen"
        assert json.dumps(res.json()) != ""
