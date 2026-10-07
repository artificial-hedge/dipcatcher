"""Streaming polish: grace-window keepalive on every create-stream surface.

POST /v1/chat/completions, /v1/completions, /v1/responses, and /v1/messages
used to run the gated call synchronously inside the 200 response — a slow
backend held the connection silent the whole generation. Now a ``stream=true``
call under ``sse_keepalive_s`` pipes work through a grace window: fast calls
are unchanged, slow calls keep alive with dialect-correct frames (``: keepalive``
comments on the OpenAI legs, unnumbered ``ping`` events on the Anthropic leg —
both consume no ``id:`` slot so ``Last-Event-ID`` resume indexing survives),
and a backend fault lands as an in-band error frame in grammar instead of a
hang. The worker runs inside ``contextvars.copy_context()`` so auth-key
attribution and metering survive the thread hop.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from starlette.testclient import TestClient

import fx1.serve.api as api_mod


class _EchoBackend:
    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        return f"clean:{messages[-1]['content']}"


class _SlowBackend(_EchoBackend):
    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        time.sleep(0.4)
        return "slow-answer"


class _FailBackend(_EchoBackend):
    def complete(self, messages: list[dict[str, str]], *, sampling: Any = None) -> str:
        time.sleep(0.4)
        raise RuntimeError("engine died mid-generation")


def _app(backend: Any, keepalive_s: float = 0.05) -> TestClient:
    return TestClient(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: backend,
            sse_keepalive_s=keepalive_s,
        )
    )


_CHAT = {"model": "fx1", "messages": [{"role": "user", "content": "x"}], "stream": True}
_MSG = {
    "model": "fx1",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "x"}],
    "stream": True,
}
_RESP = {"model": "fx1", "input": "x", "stream": True}
_LEGACY = {"model": "fx1", "prompt": "x", "stream": True}


def test_messages_keepalive_pings_then_grammar() -> None:
    """Past the grace window: unnumbered `ping` frames, then the normal
    Anthropic event sequence — pings consume no id slot."""
    r = _app(_SlowBackend()).post("/v1/messages", json=_MSG)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    # the completion fingerprint is only in-band on the keepalived leg
    assert "x-fx1-completion-id" not in r.headers
    blocks = [b for b in r.text.split("\n\n") if b.strip()]

    def _event_name(block: str) -> str | None:
        return next(
            (ln.split(":", 1)[1].strip() for ln in block.splitlines() if ln.startswith("event:")),
            None,
        )

    names = [_event_name(b) for b in blocks]
    # keepalive pings precede the grammar's message_start
    assert names[0] == "ping"
    ms_idx = names.index("message_start")
    assert all(n == "ping" for n in names[:ms_idx])
    assert names[-1] == "message_stop"
    # keepalive pings carry no `id:` slot — a `Last-Event-ID` resume
    # counts only the numbered grammar frames, unchanged by keepalives
    # (the grammar's own in-sequence `ping` after message_start does
    # carry an id — only the keepalive frames are unnumbered)
    assert not any("id:" in b for b in blocks[:ms_idx])
    numbered = [b for b in blocks if b.startswith("id:")]
    ids = [int(b.splitlines()[0].split(":", 1)[1]) for b in numbered]
    assert ids == list(range(len(numbered)))
    # the completion id survives in-band as msg_<cid> on message_start
    ms = next(b for b in blocks if _event_name(b) == "message_start")
    assert '"id":"msg_' in ms
    assert "slow-answer" in r.text


def test_messages_grace_window_fast_path_unchanged() -> None:
    """Inside the grace window the answer is the ordinary contract —
    grammar ping only, headers include the completion id."""
    r = _app(_EchoBackend(), keepalive_s=5.0).post("/v1/messages", json=_MSG)
    assert r.status_code == 200
    assert "x-fx1-completion-id" in r.headers
    assert '"type":"ping"' in r.text
    assert "message_start" in r.text
    assert "message_stop" in r.text
    assert "clean:x" in r.text


def test_messages_mid_window_fault_is_inband_error_event() -> None:
    r = _app(_FailBackend()).post("/v1/messages", json=_MSG)
    assert r.status_code == 200
    assert "event: ping" in r.text
    assert "event: error" in r.text
    assert '"type":"error"' in r.text
    assert '"type":"api_error"' in r.text
    assert "message_start" not in r.text


def test_chat_keepalive_comments_then_chunks() -> None:
    r = _app(_SlowBackend()).post("/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert "x-fx1-completion-id" not in r.headers
    lines = r.text.splitlines()
    assert ": keepalive" in lines
    first_keepalive = lines.index(": keepalive")
    first_chunk = next(i for i, ln in enumerate(lines) if ln == "id: 0")
    assert first_keepalive < first_chunk
    assert "chatcmpl-" in r.text
    assert "slow-answer" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_chat_grace_window_fast_path_unchanged() -> None:
    r = _app(_EchoBackend(), keepalive_s=5.0).post("/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert "x-fx1-completion-id" in r.headers
    assert ": keepalive" not in r.text
    assert "clean:x" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_chat_mid_window_fault_is_inband_error_frame() -> None:
    r = _app(_FailBackend()).post("/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert ": keepalive" in r.text
    assert '"type":"server_error"' in r.text
    assert "engine died mid-generation" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_completions_legacy_keepalive() -> None:
    r = _app(_SlowBackend()).post("/v1/completions", json=_LEGACY)
    assert r.status_code == 200
    assert ": keepalive" in r.text
    assert "slow-answer" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")


def test_responses_keepalive_comments_then_events() -> None:
    r = _app(_SlowBackend()).post("/v1/responses", json=_RESP)
    assert r.status_code == 200
    assert ": keepalive" in r.text
    assert "response.created" in r.text
    assert "response.completed" in r.text
    # keepalive comments consume no id slot — resume counts only real events
    assert "id: 0" in r.text


def test_responses_mid_window_fault_is_event_error() -> None:
    r = _app(_FailBackend()).post("/v1/responses", json=_RESP)
    assert r.status_code == 200
    assert "event: error" in r.text
    assert '"error"' in r.text
    assert "response.completed" not in r.text


def test_grace_window_fault_stays_json() -> None:
    """A refusal inside the grace window never opens an SSE stream — the
    wire gets the same JSON error envelope as the synchronous path."""
    fast_fail = TestClient(
        api_mod.create_app(
            backend_resolver=lambda *a, **k: _FailBackend(),
            sse_keepalive_s=60.0,
        )
    )
    r = fast_fail.post("/v1/chat/completions", json=_CHAT)
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["type"] == "server_error"


def test_non_stream_path_unchanged_json() -> None:
    r = _app(_SlowBackend()).post("/v1/chat/completions", json={**_CHAT, "stream": False})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["choices"][0]["message"]["content"] == "slow-answer"
    assert "x-fx1-completion-id" in r.headers


def test_managed_key_attribution_survives_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grace pipe copies request contextvars into the worker — a
    managed key on a keepalived stream is still attributed and metered."""
    monkeypatch.setenv(api_mod._API_KEY_ENV, "k3y-material")
    tc = _app(_SlowBackend())
    root = {"X-API-Key": "k3y-material"}
    mint = tc.post("/harness/keys", json={"name": "svc"}, headers=root)
    assert mint.status_code == 201
    kid = mint.json()["id"]
    mkey = mint.json()["key"]
    usage_before = tc.get(f"/harness/keys/{kid}/usage", headers=root).json()
    r = tc.post("/v1/chat/completions", json=_CHAT, headers={"X-API-Key": mkey})
    assert r.status_code == 200
    assert ": keepalive" in r.text
    usage_after = tc.get(f"/harness/keys/{kid}/usage", headers=root).json()
    assert usage_after["served"]["calls"] == usage_before["served"]["calls"] + 1
