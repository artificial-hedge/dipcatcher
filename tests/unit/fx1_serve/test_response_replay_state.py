"""SYNTHETIC idempotent replay cases preserve live response state."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import openai_compat
from fx1.serve.openai_compat import OpenAIEnvelopeStore


class CapturedResponse:
    def __init__(self, content: Any, *, headers: dict[str, str], **kwargs: Any) -> None:
        self.content = content
        self.headers = headers


def _invoke_replay(
    store: OpenAIEnvelopeStore,
    cached: dict[str, Any],
    *,
    stored: bool = True,
    stream: bool = False,
) -> tuple[CapturedResponse, dict[str, Any]]:
    source = Path(openai_compat.__file__).with_name("api.py")
    tree = ast.parse(source.read_text())
    route = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "openai_responses"
    )
    route.decorator_list = []
    captured_stream: dict[str, Any] = {}

    def stream_frames(*args: Any, **kwargs: Any) -> Any:
        captured_stream.update(kwargs)
        return iter(["synthetic stream frame"])

    bindings: dict[str, Any] = dict(
        Depends=lambda dependency: None,
        Header=lambda **kwargs: None,
        slot=lambda: None,
        _resume_skip=lambda *args, **kwargs: 0,
        _body_fp=lambda body, **kwargs: "synthetic fingerprint",
        _idem_lookup=lambda *args: ("synthetic key", types.SimpleNamespace(envelope=cached)),
        _openai_idem_claim=lambda: None,
        openai_idem_store=object(),
        envelope_store=store,
        _completion_receipt_sha=lambda completion_id: "a" * 64,
        JSONResponse=CapturedResponse,
        StreamingResponse=CapturedResponse,
        response_output_pieces=lambda envelope: ("answer", "item_review", None, None, None),
        _responses_sse=stream_frames,
    )
    exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), "exec"), bindings)
    response = bindings["openai_responses"](
        types.SimpleNamespace(store=stored, stream=stream),
        types.SimpleNamespace(headers={}),
        idempotency_key="synthetic key",
        last_event_id=None,
    )
    return response, captured_stream


def _envelope(status: str, *, marker: str) -> dict[str, Any]:
    return {
        "id": "resp_replay_review",
        "object": "response",
        "status": status,
        "created_at": 1,
        "output": [] if status in {"queued", "in_progress"} else [{"text": marker}],
        "_fx1_completion_id": "completion_" + marker,
    }


@pytest.mark.parametrize(
    ("cached_status", "live_status"),
    [
        ("queued", "in_progress"),
        ("queued", "cancelled"),
        ("queued", "completed"),
        ("completed", "cancelled"),
    ],
)
def test_cached_replay_cannot_replace_newer_live_response(
    cached_status: str, live_status: str
) -> None:
    store = OpenAIEnvelopeStore()
    cached = _envelope(cached_status, marker="cached")
    live = _envelope(live_status, marker="live")
    items = [{"id": "input_live"}]
    store.put(live, items={"input_items": items})
    response, _ = _invoke_replay(store, cached)
    assert store.get(live["id"]) == live
    assert store.get_items(live["id"], "input_items") == items
    assert response.content == {
        key: value for key, value in live.items() if not key.startswith("_fx1_")
    }
    assert response.headers["X-Fx1-Completion-Id"] == "completion_live"
    assert response.headers["X-Fx1-Idempotent-Replay"] == "true"
    assert cached["status"] == cached_status


@pytest.mark.parametrize("previously_deleted", [False, True])
def test_cached_replay_preserves_absent_record_rehydration(previously_deleted: bool) -> None:
    store = OpenAIEnvelopeStore()
    cached = _envelope("completed", marker="cached")
    if previously_deleted:
        store.put(cached, items={"input_items": [{"id": "discarded_input"}]})
        assert store.delete(cached["id"])
    response, _ = _invoke_replay(store, cached)
    assert store.get(cached["id"]) == cached
    assert store.get_items(cached["id"], "input_items") == []
    assert response.content["status"] == "completed"
    assert response.headers["X-Fx1-Completion-Id"] == "completion_cached"


def test_store_false_replay_leaves_retrieval_index_empty() -> None:
    store = OpenAIEnvelopeStore()
    cached = _envelope("completed", marker="cached")
    response, _ = _invoke_replay(store, cached, stored=False)
    assert len(store) == 0
    assert response.content["status"] == "completed"


def test_replay_still_refreshes_retrieval_eviction_order() -> None:
    store = OpenAIEnvelopeStore(cap=2)
    cached = _envelope("completed", marker="cached")
    store.put(cached)
    store.put({"id": "resp_older_other", "status": "completed"})
    _invoke_replay(store, cached)
    store.put({"id": "resp_new", "status": "queued"})
    assert store.get(cached["id"]) == cached
    assert store.get("resp_older_other") is None
    assert store.get("resp_new") is not None


def test_replay_preserves_cache_only_raw_usage_for_streaming() -> None:
    store = OpenAIEnvelopeStore()
    live = _envelope("completed", marker="live")
    live["model"] = "fx1"
    cached = {**live, "_fx1_usage": {"prompt_tokens": 2, "completion_tokens": 3}}
    store.put(live)
    response, stream = _invoke_replay(store, cached, stream=True)
    assert stream["usage"] == cached["_fx1_usage"]
    assert stream["final_status"] == "completed"
    assert response.headers["X-Fx1-Completion-Id"] == "completion_live"
