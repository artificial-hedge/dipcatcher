"""Adversarial probes pinned by the line-level audit of src/fx1/serve/api.py.

The defect class found in this lane: dialect-translate refusal paths
built error envelopes from ``str(ValidationError)`` — pydantic renders
``input_value`` reprs into that string, so ``fx1.byok.api_key`` (and
userinfo inside ``byok.base_url``) echoed into the body. The reachable
site was ``batch_line_body``: batch input-file lines are raw dicts, so a
credential-bearing ``fx1.byok`` inside a line that fails chat/responses
validation journaled into the durable output file. The same render was
hardened at the three dialect routes (``anthropic_messages``,
``_abatch_row_fault``, ``openai_completions``) where mirrored upstream
validators make it unreachable from the wire today — schema drift would
silently reopen the leak otherwise.

The remaining probes pin the audited boundaries that already hold:
request-layer validation redacts credential inputs, ambiguous or
oversized ingress refuses before any route runs.

SYNTHETIC, deterministic, loopback-only. Every "credential" below is a
forged marker string — the assertions check it never appears; no real
secret exists in this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fx1.serve import conv_audit as audit

# Distinctive forged markers — a leak assertion that itself contains no
# real credential material (house convention: forged keys use the
# fx1k_* marker shape). pydantic's input_value repr shows ~25 chars of
# head and tail, so the marker stays short and rides the last key of
# the body dict — it renders in full on the unfixed code path.
_SECRET = "fx1k_forged"
_USERINFO = "https://probeuser:probe-pw-marker@probe.invalid"
_TERMINAL_BATCH = {"completed", "failed", "expired", "cancelled"}


def _byok(**overrides: Any) -> dict[str, Any]:
    # api_key last: the input repr's tail window lands on it.
    byok = {"base_url": "https://probe.invalid", "model": "fx1", "api_key": _SECRET}
    byok.update(overrides)
    return byok


def _chat_line_body(**overrides: Any) -> dict[str, Any]:
    """A batch line body that fails the chat model at ``tool_choice``
    coherence — the model-level check renders the whole input dict."""
    body = {
        "model": "fx1",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "auto",
        "fx1": {"byok": _byok()},
    }
    body.update(overrides)
    return body


def _responses_line_body(**overrides: Any) -> dict[str, Any]:
    """A batch line body that fails the responses model at
    ``reasoning.effort`` — same whole-input echo site."""
    body = {
        "model": "fx1",
        "input": "hi",
        "reasoning": {"effort": "bogus"},
        "fx1": {"byok": _byok()},
    }
    body.update(overrides)
    return body


def _poll(client: Any, path: str, done: set[str], status_key: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while True:
        rec = client.get(path).json()
        if rec[status_key] in done:
            return rec
        assert time.monotonic() < deadline, f"{path} never reached {sorted(done)}"
        time.sleep(0.02)


def _journals_clean(state_dir: Path, *needles: str) -> None:
    """No marker may appear in any journaled byte stream."""
    for journal in state_dir.glob("*.jsonl"):
        for needle in needles:
            assert needle.encode() not in journal.read_bytes(), journal.name


def _run_batch(client: Any, line_body: dict[str, Any]) -> str:
    """Submit one errored batch line through the full /v1/files +
    /v1/batches pipeline; return the output-file content."""
    line = {
        "custom_id": "c1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": line_body,
    }
    upload = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert upload.status_code == 200, upload.text
    created = client.post(
        "/v1/batches",
        json={
            "input_file_id": upload.json()["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert created.status_code == 200, created.text
    rec = _poll(client, f"/v1/batches/{created.json()['id']}", _TERMINAL_BATCH, "status")
    assert rec["status"] == "completed"
    content = client.get(f"/v1/files/{rec['output_file_id']}/content")
    assert content.status_code == 200
    return content.text


# ---------------------------------------------------------------------------
# defect: str(ValidationError) journaled request input into output rows
# ---------------------------------------------------------------------------


def test_batch_chat_line_never_journals_byok() -> None:
    """The journaled output row must carry the verdict msg only — before
    the fix, ``input_value`` embedded the line body, api_key included."""
    with audit._audit_context():
        state = audit._temporary_directory()
        client, _ = audit._client({audit._MODEL: audit._StubBackend}, state_dir=state)
        content = _run_batch(client, _chat_line_body())
        assert _SECRET not in content
        row = json.loads(content.strip().splitlines()[0])
        assert row["custom_id"] == "c1"
        assert row["response"]["status_code"] == 400
        message = row["response"]["body"]["error"]["message"]
        # the refusal still names the violated rule — sanitization keeps
        # the verdict text, dropping only the echoed input values.
        assert "tool_choice" in message
        assert "invalid request body" in message
        _journals_clean(state, _SECRET)


def test_batch_responses_line_never_journals_byok() -> None:
    """The /v1/responses endpoint's own model validators fail the same
    way — the output row must not carry the credential either."""
    with audit._audit_context():
        state = audit._temporary_directory()
        client, _ = audit._client({audit._MODEL: audit._StubBackend}, state_dir=state)
        line = {
            "custom_id": "c1",
            "method": "POST",
            "url": "/v1/responses",
            "body": _responses_line_body(),
        }
        upload = client.post(
            "/v1/files",
            files={"file": ("in.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
            data={"purpose": "batch"},
        )
        assert upload.status_code == 200, upload.text
        created = client.post(
            "/v1/batches",
            json={
                "input_file_id": upload.json()["id"],
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            },
        )
        assert created.status_code == 200, created.text
        rec = _poll(client, f"/v1/batches/{created.json()['id']}", _TERMINAL_BATCH, "status")
        assert rec["status"] == "completed"
        content = client.get(f"/v1/files/{rec['output_file_id']}/content").text
        assert _SECRET not in content
        row = json.loads(content.strip().splitlines()[0])
        assert row["response"]["status_code"] == 400
        assert "reasoning.effort" in row["response"]["body"]["error"]["message"]
        _journals_clean(state, _SECRET)


def test_batch_line_userinfo_url_never_journals() -> None:
    """A byok.base_url carrying userinfo fails its field validator — the
    pasted credentials must not echo into the journaled output row."""
    with audit._audit_context():
        state = audit._temporary_directory()
        client, _ = audit._client({audit._MODEL: audit._StubBackend}, state_dir=state)
        content = _run_batch(client, _chat_line_body(fx1={"byok": _byok(base_url=_USERINFO)}))
        for marker in (_USERINFO, "probe-pw-marker", "probeuser"):
            assert marker not in content
        _journals_clean(state, "probe-pw-marker", "probeuser")


# ---------------------------------------------------------------------------
# boundaries that already hold — request layer, ingress
# ---------------------------------------------------------------------------


def test_request_layer_422_redacts_credential_field_input() -> None:
    """A request-layer failure inside ``fx1.byok`` must redact the
    rejected value — never echo the credential field's input."""
    with audit._audit_context():
        client, _ = audit._client({audit._MODEL: audit._StubBackend})
        resp = client.post(
            "/v1/messages",
            json={
                "model": "fx1",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "fx1": {"byok": _byok(base_url=_USERINFO)},
            },
        )
        assert resp.status_code == 422, resp.text
        for marker in (_USERINFO, "probe-pw-marker", "probeuser"):
            assert marker not in resp.text
        assert "byok.base_url" in resp.json()["error"]["message"]


def test_ambiguous_credentials_refused_before_route() -> None:
    """Two different credential headers on one request — the ingress
    layer refuses rather than picking one to authenticate."""
    with audit._audit_context():
        client, _ = audit._client({audit._MODEL: audit._StubBackend}, api_key="fx1k_forged")
        resp = client.post(
            "/v1/messages",
            json={"model": "fx1", "max_tokens": 8, "messages": []},
            headers={"X-API-Key": "fx1k_forged", "Authorization": "Bearer fx1k_forged"},
        )
        assert resp.status_code == 400
        assert "authentication header" in resp.text


def test_oversized_body_refused_413() -> None:
    """The 1 MiB entity cap is enforced pre-route, before any body parse."""
    with audit._audit_context():
        client, _ = audit._client({audit._MODEL: audit._StubBackend})
        resp = client.post(
            "/v1/messages",
            content=b'{"model":"fx1","max_tokens":8,"messages":'
            b'[{"role":"user","content":"' + b"x" * (1 << 21) + b'"}]}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413


def test_native_chat_tool_choice_refusal_is_a_clean_422() -> None:
    """Same violated rule on the native dialect: the chat body fails at
    the request layer, where ``loc: msg`` joins carry no input values."""
    with audit._audit_context():
        client, _ = audit._client({audit._MODEL: audit._StubBackend})
        resp = client.post("/v1/chat/completions", json=_chat_line_body())
        assert resp.status_code == 422, resp.text
        assert _SECRET not in resp.text
