"""store_audit — adversarial probes on the ``store=`` persistence boundary.

The claim under test: ``store`` is a retrieval-index knob, never an
evidence knob. ``store=false`` on ``/v1/chat/completions`` and
``/v1/responses`` keeps the minted envelope out of the
``GET``/``DELETE``/list/subresource tier while the call still executes,
still bills the acting credential, and still lands in the completion
log; surfaces without a retrieval twin (``/v1/messages``,
``/v1/completions``) pin the flag inert at translation. This battery
measures that boundary end-to-end — retrieval, subresources, lists,
idempotency, restart durability, batch per-line, and metering — and
pins the cases where the boundary is deliberately asymmetric.

Coverage map:

- *Chat boundary* — a ``store:false`` completion mints a ``chatcmpl-``
  id that never resolves: ``GET``, ``/messages``, list, ``POST``
  update, and ``DELETE`` all answer 404 while ``store:true`` and the
  absent-default pin. The response body is byte-identical to the stored
  twin minus volatile fields; ``n>1`` fans out unstored too.
- *Responses boundary* — ``store:false`` mints a ``resp_`` id that
  answers 404 on ``GET``, ``?stream=true`` replay, ``input_items``,
  ``DELETE`` and ``cancel``; the object still echoes ``store:false``.
  ``previous_response_id`` off an unstored parent fails closed
  ``400 previous_response_not_found`` (inline and at background submit);
  ``conversation`` binding still appends the turn — the conv is its own
  store, so the flag governs only the response's own index entry.
- *Surfaces with no store knob* — ``/v1/messages`` and
  ``/v1/completions`` force ``store=False`` at translation: an explicit
  ``store:true`` extra is tolerated but inert, the derived
  ``chatcmpl-`` twin never enters the index, and ``GET
  /v1/completions/{id}`` has no route at all (catch-all 404).
- *Streaming* — ``store:false`` streams complete with the correct
  terminal frame (``[DONE]`` for chat/legacy, ``response.completed``
  for responses, ``message_stop`` for Anthropic) and persist nothing.
- *Background* — ``background=true`` + ``store=false`` is a correct
  refusal (``400 background_requires_store``): a background response is
  reachable only through the index, so the request fails fast rather
  than computing an unreachable result. ``background``+``stream`` runs
  the normal stream — the flag is then honored, not refused.
- *Idempotency* — the ``Idempotency-Key`` ledger is independent of the
  retrieval index: a keyed ``store:false`` call pins a replay record,
  a retry replays it (no re-execution, ``X-Fx1-Idempotent-Replay``) and
  the id still answers 404 — the replay never repins a ``store=false``
  envelope. The same ledger serves chat and responses (cross-surface
  key reuse fails closed 409), and the inert-store surfaces replay
  normally.
- *List boundary* — unstored calls never surface in
  ``GET /v1/chat/completions`` under any order/filter/window; a
  conversation's item list is the honest exception (the conv keeps the
  ``store:false`` turn's items by design); ``/v1/responses`` and
  ``/v1/messages`` have no list routes (catch-all 404 in each
  surface's own grammar).
- *Restart durability* — the retrieval index is a documented
  in-memory fetch cache: even ``store:true`` chat/response envelopes
  answer 404 after a ``--state-dir`` restart, while the journaled
  stores — conversations (items appended by an unstored turn
  included), the idem ledgers, key meters, eval runs, batches — all
  survive. The completion log is an in-process ring: it empties on
  restart like the index.
- *Batch per-line* — ``store`` inside a ``/v1/batches`` input line is
  honored per line: a ``store:false`` line's result still lands in the
  output file but its minted id answers 404, while a default-stored
  sibling line retrieves normally.
- *Eval records* — eval specs and runs live in the journaled eval
  store; there is no ``store`` knob — records are always retrievable
  and survive restart.
- *Metering* — ``store:false`` calls bill exactly like stored ones:
  per-key ``uses``/``tokens_used`` advance identically and the
  completion-log row carries the acting credential's fingerprint. The
  flag governs retrieval, never evidence.

No defects found in this lane — the boundary holds as designed. The
measured asymmetries (in-memory retrieval index and completion log vs
journaled conversations/idempotency/eval/batch/key stores; a
``store=false`` conversation turn kept by the container) are pinned as
the documented contract.

Sealed ``store_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _temporary_directory,
)

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["store_audit", "store_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_MODEL = "byok"
_ANTHROPIC_H = {"anthropic-version": "2023-06-01"}
_CHAT_PATH = "/v1/chat/completions"
_RESP_PATH = "/v1/responses"
_MSG_PATH = "/v1/messages"
_LEGACY_PATH = "/v1/completions"
_CONV_PATH = "/v1/conversations"
_FILES_PATH = "/v1/files"
_BATCHES_PATH = "/v1/batches"
_EVALS_PATH = "/v1/evals"
_LEDGER_PATH = "/harness/completions"
_REPLAY_H = "x-fx1-idempotent-replay"
_IDEM = "Idem-Store-1"


# ---------------------------------------------------------------------------
# Stub backend + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn and
    counts executions so a replay probe can prove no second spend."""

    def __init__(self, model: str = "store-stub-0") -> None:
        self._model = model
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        return f"stub:{messages[-1]['content']}"

    def close(self) -> None:
        pass


def _backends(stub: _StubBackend | None = None) -> dict[str, Any]:
    stub = stub or _StubBackend()
    return {
        "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
        "local_fx1": lambda **k: _StubBackend("ckpt-stub"),
        "byok": lambda **k: stub,
    }


def _client(
    backend_map: dict[str, Any] | None = None,
    *,
    api_key: str | None = _ROOT,
    state_dir: Path | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = None
    import os

    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
            backend_resolver=lambda name, *a, **k: (backend_map or _backends())[name](**k),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
        )
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key
    resources = _RESOURCES.get()
    resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
    client = TestClient(app, raise_server_exceptions=False)
    resources.callback(client.close)
    resources.enter_context(client)
    return client, api_mod


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _ah(auth: str | None) -> dict[str, str]:
    return {**_ANTHROPIC_H, **_h(auth)}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    """Mint a managed key under the env root → (raw material, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=_h(_ROOT))
    assert r.status_code in (200, 201), r.text
    return str(r.json()["key"]), str(r.json()["id"])


def _card(client: TestClient, key_id: str) -> dict[str, Any]:
    r = client.get(f"/harness/keys/{key_id}/usage", headers=_h(_ROOT))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _uses(client: TestClient, key_id: str) -> int:
    return int(_card(client, key_id)["uses"])


def _tokens(client: TestClient, key_id: str) -> int:
    return int(_card(client, key_id)["tokens_used"])


def _chat(client: TestClient, auth: str, text: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return client.post(_CHAT_PATH, json=body, headers=_h(auth))


def _respond(client: TestClient, auth: str, text: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "input": text}
    body.update(extra)
    return client.post(_RESP_PATH, json=body, headers=_h(auth))


def _legacy(client: TestClient, auth: str, prompt: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "prompt": prompt}
    body.update(extra)
    return client.post(_LEGACY_PATH, json=body, headers=_h(auth))


def _message(client: TestClient, auth: str, text: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 64}
    body.update(extra)
    return client.post(_MSG_PATH, json=body, headers=_ah(auth))


def _conv(client: TestClient, auth: str, **body: Any) -> dict[str, Any]:
    r = client.post(_CONV_PATH, json=body, headers=_h(auth))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _conv_items(client: TestClient, auth: str, cid: str, **params: Any) -> dict[str, Any]:
    r = client.get(f"{_CONV_PATH}/{cid}/items", headers=_h(auth), params=params)
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _list_ids(resp: Any) -> set[str]:
    if resp.status_code != 200:
        return set()
    return {str(item["id"]) for item in resp.json().get("data", [])}


def _wait_response(client: TestClient, auth: str, rid: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(f"{_RESP_PATH}/{rid}", headers=_h(auth)).json()
        if rec.get("status") in ("completed", "failed", "cancelled", "incomplete"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"response {rid} never reached a terminal state")


def _wait_batch(client: TestClient, auth: str, batch_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(f"{_BATCHES_PATH}/{batch_id}", headers=_h(auth)).json()
        if rec.get("status") in ("completed", "failed", "cancelled", "expired"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} never reached a terminal state")


def _wait_eval_run(client: TestClient, auth: str, eval_id: str, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(
            f"{_EVAL_PATH}/{eval_id}/runs/{run_id}", headers=_h(auth)
        ).json()
        if rec.get("status") in ("completed", "failed", "canceled", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"eval run {run_id} never reached a terminal state")


_EVAL_PATH = _EVALS_PATH


def _eval_spec(client: TestClient, auth: str, name: str) -> str:
    r = client.post(
        _EVALS_PATH,
        json={
            "name": name,
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 11, "backend": "byok"},
            },
            "metadata": {"probe": name},
        },
        headers=_h(auth),
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def _upload_file(client: TestClient, auth: str, name: str, body: bytes, purpose: str) -> str:
    r = client.post(
        _FILES_PATH,
        files={"file": (name, body)},
        data={"purpose": purpose},
        headers=_h(auth),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _batch_line(custom_id: str, url: str, body: dict[str, Any]) -> bytes:
    line = {"custom_id": custom_id, "method": "POST", "url": url, "body": body}
    return (json.dumps(line) + "\n").encode()


def _ledger_row(client: TestClient, cid: str) -> dict[str, Any] | None:
    r = client.get(f"{_LEDGER_PATH}/{cid}", headers=_h(_ROOT))
    if r.status_code != 200:
        return None
    out: dict[str, Any] = r.json()
    return out


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error")
    return dict(err) if isinstance(err, dict) else {}


def _sse_data_frames(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("data:"))


_VOLATILE = frozenset({"id", "created", "created_at", "store"})


def _strip_volatile(node: Any) -> Any:
    """Deep-copy minus volatile keys at every level — two runs of the
    same request must be byte-identical except ids/timestamps/the flag."""
    if isinstance(node, dict):
        return {k: _strip_volatile(v) for k, v in node.items() if k not in _VOLATILE}
    if isinstance(node, list):
        return [_strip_volatile(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Chat boundary — the minted id never resolves under store=false
# ---------------------------------------------------------------------------


def _probe_chat_boundary(results: dict[str, bool]) -> None:
    """``store:false`` on ``/v1/chat/completions`` mints a ``chatcmpl-``
    id that never enters the retrieval index: GET, the ``/messages``
    subresource, the list surface, metadata update and DELETE all answer
    the same 404, while ``store:true`` and the absent default pin. The
    wire body is identical to the stored twin minus volatile fields."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    unstored = _chat(client, key, "chat-body", store=False)
    results["chat_store_false_mints_id"] = unstored.status_code == 200 and str(
        unstored.json()["id"]
    ).startswith("chatcmpl-")
    stored = _chat(client, key, "chat-body", store=True)
    cid = str(unstored.json()["id"])
    got = client.get(f"{_CHAT_PATH}/{cid}", headers=_h(key))
    results["chat_store_false_get_404"] = got.status_code == 404
    results["chat_store_false_get_404_enveloped"] = _err(got).get("code") == "not_found"
    results["chat_store_false_messages_404"] = (
        client.get(f"{_CHAT_PATH}/{cid}/messages", headers=_h(key)).status_code == 404
    )
    listed = client.get(_CHAT_PATH, headers=_h(key), params={"limit": 100})
    results["chat_store_false_list_excludes"] = cid not in _list_ids(listed)
    updated = client.post(f"{_CHAT_PATH}/{cid}", json={"metadata": {"k": "v"}}, headers=_h(key))
    results["chat_store_false_update_404"] = updated.status_code == 404
    deleted = client.delete(f"{_CHAT_PATH}/{cid}", headers=_h(key))
    results["chat_store_false_delete_404_enveloped"] = (
        deleted.status_code == 404 and _err(deleted).get("code") == "not_found"
    )

    sid = str(stored.json()["id"])
    results["chat_store_true_get_200"] = (
        client.get(f"{_CHAT_PATH}/{sid}", headers=_h(key)).status_code == 200
    )
    msgs = client.get(f"{_CHAT_PATH}/{sid}/messages", headers=_h(key))
    results["chat_store_true_messages_200"] = (
        msgs.status_code == 200 and len(msgs.json().get("data", [])) >= 1
    )
    results["chat_store_true_listed"] = sid in _list_ids(
        client.get(_CHAT_PATH, headers=_h(key), params={"limit": 100})
    )
    defaulted = _chat(client, key, "chat-default")
    did = str(defaulted.json()["id"])
    results["chat_store_default_pins"] = (
        client.get(f"{_CHAT_PATH}/{did}", headers=_h(key)).status_code == 200
    )

    results["chat_store_false_body_parity"] = _strip_volatile(unstored.json()) == _strip_volatile(
        stored.json()
    )

    header_cid = unstored.headers.get("x-fx1-completion-id")
    row = _ledger_row(client, str(header_cid)) if header_cid else None
    results["chat_store_false_completion_header_links_log"] = (
        header_cid is not None and row is not None and row.get("ok") is True
    )
    results["chat_store_false_receipt_sha_issued"] = bool(
        unstored.headers.get("x-fx1-receipt-sha256")
    )
    n2 = _chat(client, key, "chat-n2", store=False, n=2)
    results["chat_n2_store_false_unpinned"] = n2.status_code == 200 and (
        client.get(f"{_CHAT_PATH}/{n2.json()['id']}", headers=_h(key)).status_code == 404
    )
    results["chat_never_issued_id_404"] = (
        client.get(f"{_CHAT_PATH}/chatcmpl-{'0' * 32}", headers=_h(key)).status_code == 404
    )


# ---------------------------------------------------------------------------
# Responses boundary — the same knob, deeper surface
# ---------------------------------------------------------------------------


def _probe_responses_boundary(results: dict[str, bool]) -> None:
    """``store:false`` on ``/v1/responses``: the ``resp_`` id 404s on
    GET, ``?stream=true`` replay, ``input_items``, DELETE and cancel;
    the object still echoes the honored flag. ``previous_response_id``
    off an unstored parent fails closed; ``conversation`` keeps the turn
    (the conv is its own store)."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    unstored = _respond(client, key, "resp-body", store=False)
    results["responses_store_false_mints_id"] = unstored.status_code == 200 and str(
        unstored.json()["id"]
    ).startswith("resp_")
    rid = str(unstored.json()["id"])
    results["responses_store_false_echoes_flag"] = unstored.json().get("store") is False
    got = client.get(f"{_RESP_PATH}/{rid}", headers=_h(key))
    results["responses_store_false_get_404"] = got.status_code == 404
    results["responses_store_false_get_404_enveloped"] = _err(got).get("code") == "not_found"
    results["responses_store_false_stream_replay_404"] = (
        client.get(f"{_RESP_PATH}/{rid}", params={"stream": "true"}, headers=_h(key)).status_code
        == 404
    )
    results["responses_store_false_input_items_404"] = (
        client.get(f"{_RESP_PATH}/{rid}/input_items", headers=_h(key)).status_code == 404
    )
    deleted = client.delete(f"{_RESP_PATH}/{rid}", headers=_h(key))
    results["responses_store_false_delete_404_enveloped"] = (
        deleted.status_code == 404 and _err(deleted).get("code") == "not_found"
    )
    results["responses_store_false_cancel_404"] = (
        client.post(f"{_RESP_PATH}/{rid}/cancel", headers=_h(key)).status_code == 404
    )

    stored = _respond(client, key, "resp-body", store=True)
    rid_on = str(stored.json()["id"])
    results["responses_store_true_echoes_flag"] = stored.json().get("store") is True
    results["responses_store_true_get_200"] = (
        client.get(f"{_RESP_PATH}/{rid_on}", headers=_h(key)).status_code == 200
    )
    replay = client.get(f"{_RESP_PATH}/{rid_on}", params={"stream": "true"}, headers=_h(key))
    results["responses_store_true_stream_replay_streams"] = (
        replay.status_code == 200 and "response.completed" in replay.text
    )
    items = client.get(f"{_RESP_PATH}/{rid_on}/input_items", headers=_h(key))
    results["responses_store_true_input_items_200"] = (
        items.status_code == 200 and len(items.json().get("data", [])) >= 1
    )
    results["responses_store_false_body_parity"] = _strip_volatile(
        unstored.json()
    ) == _strip_volatile(stored.json())

    chained_bad = _respond(client, key, "child", previous_response_id=rid)
    results["responses_prev_off_unstored_parent_400"] = chained_bad.status_code == 400
    results["responses_prev_off_unstored_parent_enveloped"] = (
        _err(chained_bad).get("code") == "previous_response_not_found"
    )
    child = _respond(client, key, "child", previous_response_id=rid_on)
    cid_child = str(child.json()["id"])
    results["responses_prev_off_stored_parent_200"] = child.status_code == 200
    child_items = client.get(f"{_RESP_PATH}/{cid_child}/input_items", headers=_h(key))
    results["responses_chained_child_carries_history"] = (
        child_items.status_code == 200 and len(child_items.json().get("data", [])) >= 3
    )

    conv = _conv(client, key)
    conv_id = str(conv["id"])
    n_items = len(_conv_items(client, key, conv_id, limit=100)["data"])
    conv_turn = _respond(client, key, "conv-turn", store=False, conversation=conv_id)
    results["responses_conv_store_false_turn_appends"] = (
        len(_conv_items(client, key, conv_id, limit=100)["data"]) == n_items + 2
    )
    conv_rid = str(conv_turn.json()["id"])
    results["responses_conv_store_false_response_uncached"] = (
        client.get(f"{_RESP_PATH}/{conv_rid}", headers=_h(key)).status_code == 404
    )
    bad_conv = _respond(client, key, "conv-x", conversation="conv_" + "0" * 32)
    results["responses_unknown_conv_400_enveloped"] = (
        bad_conv.status_code == 400 and _err(bad_conv).get("code") == "conversation_not_found"
    )

    bg_chain = _respond(client, key, "bg-child", background=True, previous_response_id=rid)
    results["responses_bg_prev_unstored_parent_400"] = (
        bg_chain.status_code == 400 and _err(bg_chain).get("code") == "previous_response_not_found"
    )


# ---------------------------------------------------------------------------
# Surfaces with no store knob — the flag is inert at translation
# ---------------------------------------------------------------------------


def _probe_messages_inert(results: dict[str, bool]) -> None:
    """``/v1/messages`` has no retrieval twin: translation pins
    ``store=False``, so even an explicit ``store:true`` extra is
    tolerated but inert — the derived ``chatcmpl-`` id never enters the
    OpenAI index and the call never lists. Metering still records it."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    plain = _message(client, key, "msg-a")
    results["messages_completes_200"] = plain.status_code == 200 and str(
        plain.json()["id"]
    ).startswith("msg_")
    twin = "chatcmpl-" + str(plain.json()["id"]).removeprefix("msg_")
    results["messages_derived_chat_id_unretrievable"] = (
        client.get(f"{_CHAT_PATH}/{twin}", headers=_h(key)).status_code == 404
    )
    stored_extra = _message(client, key, "msg-b", store=True)
    twin_b = "chatcmpl-" + str(stored_extra.json()["id"]).removeprefix("msg_")
    results["messages_store_true_extra_inert"] = stored_extra.status_code == 200 and (
        client.get(f"{_CHAT_PATH}/{twin_b}", headers=_h(key)).status_code == 404
    )
    off_extra = _message(client, key, "msg-c", store=False)
    twin_c = "chatcmpl-" + str(off_extra.json()["id"]).removeprefix("msg_")
    results["messages_store_false_extra_inert"] = off_extra.status_code == 200 and (
        client.get(f"{_CHAT_PATH}/{twin_c}", headers=_h(key)).status_code == 404
    )
    listed = client.get(_CHAT_PATH, headers=_h(key), params={"limit": 100})
    results["messages_absent_from_chat_list"] = not ({twin, twin_b, twin_c} & _list_ids(listed))
    header_cid = plain.headers.get("x-fx1-completion-id")
    results["messages_completion_logged"] = (
        header_cid is not None and _ledger_row(client, str(header_cid)) is not None
    )


def _probe_legacy_inert(results: dict[str, bool]) -> None:
    """``/v1/completions`` tolerates ``store`` and ignores it — the
    legacy surface has no retrieval twin at all (the catch-all 404s
    ``GET /v1/completions/{id}``), so the pinned ``Idempotency-Key``
    replay is its only retrieval path."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    plain = _legacy(client, key, "leg-a")
    results["legacy_completes_200"] = plain.status_code == 200 and str(
        plain.json()["id"]
    ).startswith("cmpl-")
    twin = "chatcmpl-" + str(plain.json()["id"]).removeprefix("cmpl-")
    results["legacy_derived_chat_id_unretrievable"] = (
        client.get(f"{_CHAT_PATH}/{twin}", headers=_h(key)).status_code == 404
    )
    stored_extra = _legacy(client, key, "leg-b", store=True)
    twin_b = "chatcmpl-" + str(stored_extra.json()["id"]).removeprefix("cmpl-")
    results["legacy_store_true_extra_inert"] = stored_extra.status_code == 200 and (
        client.get(f"{_CHAT_PATH}/{twin_b}", headers=_h(key)).status_code == 404
    )
    off_extra = _legacy(client, key, "leg-c", store=False)
    twin_c = "chatcmpl-" + str(off_extra.json()["id"]).removeprefix("cmpl-")
    results["legacy_store_false_extra_inert"] = off_extra.status_code == 200 and (
        client.get(f"{_CHAT_PATH}/{twin_c}", headers=_h(key)).status_code == 404
    )
    listed = client.get(_CHAT_PATH, headers=_h(key), params={"limit": 100})
    results["legacy_absent_from_chat_list"] = not ({twin, twin_b, twin_c} & _list_ids(listed))
    missing = client.get(f"{_LEGACY_PATH}/{plain.json()['id']}", headers=_h(key))
    results["legacy_no_retrieval_twin_404_enveloped"] = (
        missing.status_code == 404 and _err(missing).get("code") == "not_found"
    )
    header_cid = plain.headers.get("x-fx1-completion-id")
    results["legacy_completion_logged"] = (
        header_cid is not None and _ledger_row(client, str(header_cid)) is not None
    )


# ---------------------------------------------------------------------------
# Streaming — the stream completes and persists nothing
# ---------------------------------------------------------------------------


def _probe_streaming(results: dict[str, bool]) -> None:
    """A ``store:false`` stream emits its full terminal grammar and
    leaves no index entry — ``[DONE]`` on chat/legacy,
    ``response.completed`` on responses, ``message_stop`` on Anthropic.
    ``stream``+``store:true`` pins the completed envelope (control)."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    chat_off = _chat(client, key, "stream-off", stream=True, store=False)
    results["chat_stream_store_false_done"] = (
        chat_off.status_code == 200 and "data: [DONE]" in chat_off.text
    )
    off_id = chat_off.json()["id"] if chat_off.status_code != 200 else None
    first_line = next((ln for ln in chat_off.text.splitlines() if ln.startswith("data: {")), "")
    stream_id = str(json.loads(first_line[5:])["id"]) if first_line else ""
    results["chat_stream_store_false_unpinned"] = bool(stream_id) and (
        client.get(f"{_CHAT_PATH}/{stream_id}", headers=_h(key)).status_code == 404
    )
    del off_id

    chat_on = _chat(client, key, "stream-on", stream=True, store=True)
    on_line = next((ln for ln in chat_on.text.splitlines() if ln.startswith("data: {")), "")
    on_id = str(json.loads(on_line[5:])["id"]) if on_line else ""
    results["chat_stream_store_true_pinned"] = bool(on_id) and (
        client.get(f"{_CHAT_PATH}/{on_id}", headers=_h(key)).status_code == 200
    )

    resp_off = _respond(client, key, "resp-stream-off", stream=True, store=False)
    results["responses_stream_store_false_completed"] = (
        resp_off.status_code == 200 and "response.completed" in resp_off.text
    )
    resp_id = ""
    for ln in resp_off.text.splitlines():
        if ln.startswith("data: {"):
            try:
                ev = json.loads(ln[5:])
            except json.JSONDecodeError:
                continue
            rid_ev = (ev.get("response") or {}).get("id") if isinstance(ev, dict) else None
            if isinstance(rid_ev, str) and rid_ev.startswith("resp_"):
                resp_id = rid_ev
    results["responses_stream_store_false_unpinned"] = bool(resp_id) and (
        client.get(f"{_RESP_PATH}/{resp_id}", headers=_h(key)).status_code == 404
    )

    msg_stream = _message(client, key, "msg-stream", stream=True)
    results["messages_stream_completes_message_stop"] = (
        msg_stream.status_code == 200 and "message_stop" in msg_stream.text
    )

    leg_stream = _legacy(client, key, "leg-stream", stream=True)
    results["legacy_stream_done"] = (
        leg_stream.status_code == 200 and "data: [DONE]" in leg_stream.text
    )


# ---------------------------------------------------------------------------
# Background — store=false is a correct refusal there
# ---------------------------------------------------------------------------


def _probe_background(results: dict[str, bool]) -> None:
    """``background=true`` needs the retrieval index — a background
    response is only reachable through it — so ``store=false`` refuses
    fast ``400 background_requires_store`` instead of computing an
    unreachable result. ``background``+``stream`` is the normal stream:
    the flag is then honored and nothing pins."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    refused = _respond(client, key, "bg-off", background=True, store=False)
    results["bg_store_false_400_requires_store"] = refused.status_code == 400
    results["bg_store_false_enveloped"] = _err(refused).get("code") == "background_requires_store"

    streamed = _respond(client, key, "bg-stream-off", background=True, stream=True, store=False)
    results["bg_stream_store_false_runs_as_stream"] = (
        streamed.status_code == 200 and "response.completed" in streamed.text
    )
    srid = ""
    for ln in streamed.text.splitlines():
        if ln.startswith("data: {"):
            try:
                ev = json.loads(ln[5:])
            except json.JSONDecodeError:
                continue
            rid_ev = (ev.get("response") or {}).get("id") if isinstance(ev, dict) else None
            if isinstance(rid_ev, str) and rid_ev.startswith("resp_"):
                srid = rid_ev
    results["bg_stream_store_false_unpinned"] = bool(srid) and (
        client.get(f"{_RESP_PATH}/{srid}", headers=_h(key)).status_code == 404
    )

    queued = _respond(client, key, "bg-on", background=True, store=True)
    results["bg_store_true_queued"] = (
        queued.status_code == 200 and queued.json().get("status") == "queued"
    )
    qrid = str(queued.json()["id"])
    results["bg_store_true_completes_and_retrieves"] = (
        _wait_response(client, key, qrid).get("status") == "completed"
    )


# ---------------------------------------------------------------------------
# Idempotency — the replay ledger is independent of the retrieval index
# ---------------------------------------------------------------------------


def _probe_idem(results: dict[str, bool]) -> None:
    """``Idempotency-Key`` pins its own replay record regardless of
    ``store``: a keyed ``store:false`` retry replays the pinned envelope
    byte-identically (no second model spend, ``X-Fx1-Idempotent-Replay``)
    and the id still 404s — the replay never repins an unstored
    envelope. Keyed streams replay and resume under ``store:false`` too;
    the same ledger serves chat and responses, so cross-surface key
    reuse conflicts 409."""
    stub = _StubBackend()
    client, _api = _client(_backends(stub))
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    kheaders = {**_h(key), "Idempotency-Key": _IDEM}
    body = {"model": _MODEL, "messages": [{"role": "user", "content": "idem-off"}], "store": False}
    calls0 = stub.calls
    first = client.post(_CHAT_PATH, json=body, headers=kheaders)
    again = client.post(_CHAT_PATH, json=body, headers=kheaders)
    results["idem_chat_store_false_replays"] = (
        again.status_code == 200
        and again.headers.get(_REPLAY_H) == "true"
        and again.json()["id"] == first.json()["id"]
    )
    results["idem_chat_store_false_no_reexecution"] = stub.calls == calls0 + 1
    results["idem_chat_store_false_still_unpinned"] = (
        client.get(f"{_CHAT_PATH}/{first.json()['id']}", headers=_h(key)).status_code == 404
    )
    conflict = client.post(
        _CHAT_PATH,
        json={"model": _MODEL, "messages": [{"role": "user", "content": "other"}], "store": False},
        headers=kheaders,
    )
    results["idem_chat_store_false_conflict_409"] = conflict.status_code == 409

    rbody = {"model": _MODEL, "input": "idem-resp", "store": False}
    rk = {**_h(key), "Idempotency-Key": "Idem-Store-2"}
    rfirst = client.post(_RESP_PATH, json=rbody, headers=rk)
    ragain = client.post(_RESP_PATH, json=rbody, headers=rk)
    results["idem_responses_store_false_replays"] = (
        ragain.status_code == 200
        and ragain.headers.get(_REPLAY_H) == "true"
        and ragain.json()["id"] == rfirst.json()["id"]
    )
    calls_after = stub.calls
    client.post(_RESP_PATH, json=rbody, headers=rk)
    results["idem_responses_store_false_no_reexecution"] = stub.calls == calls_after
    results["idem_responses_store_false_still_unpinned"] = (
        client.get(f"{_RESP_PATH}/{rfirst.json()['id']}", headers=_h(key)).status_code == 404
    )

    sbody = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "idem-s"}],
        "stream": True,
        "store": False,
    }
    sk = {**_h(key), "Idempotency-Key": "Idem-Store-3"}
    sfirst = client.post(_CHAT_PATH, json=sbody, headers=sk)
    sagain = client.post(_CHAT_PATH, json=sbody, headers=sk)
    results["idem_chat_store_false_stream_replays"] = (
        sagain.status_code == 200
        and sagain.headers.get(_REPLAY_H) == "true"
        and "data: [DONE]" in sagain.text
    )
    full_frames = _sse_data_frames(sfirst.text)
    resumed = client.post(
        _CHAT_PATH,
        json=sbody,
        headers={**sk, "Last-Event-ID": "0"},
    )
    results["idem_store_false_resume_replays_tail"] = (
        resumed.status_code == 200
        and "data: [DONE]" in resumed.text
        and _sse_data_frames(resumed.text) == full_frames - 1
    )

    lbody = {"model": _MODEL, "prompt": "idem-leg"}
    lk = {**_h(key), "Idempotency-Key": "Idem-Store-4"}
    lfirst = client.post(_LEGACY_PATH, json=lbody, headers=lk)
    lagain = client.post(_LEGACY_PATH, json=lbody, headers=lk)
    results["idem_legacy_replay_is_retrieval_path"] = (
        lagain.status_code == 200
        and lagain.headers.get(_REPLAY_H) == "true"
        and lagain.json()["id"] == lfirst.json()["id"]
    )

    mbody = {"model": _MODEL, "messages": [{"role": "user", "content": "idem-m"}], "max_tokens": 64}
    mk = {**_ah(key), "Idempotency-Key": "Idem-Store-5"}
    mfirst = client.post(_MSG_PATH, json=mbody, headers=mk)
    magain = client.post(_MSG_PATH, json=mbody, headers=mk)
    results["idem_messages_replay_works"] = (
        magain.status_code == 200
        and magain.headers.get(_REPLAY_H) == "true"
        and magain.json()["id"] == mfirst.json()["id"]
    )

    cross = client.post(
        _RESP_PATH,
        json={"model": _MODEL, "input": "x", "store": True},
        headers=kheaders,
    )
    results["idem_shared_ledger_cross_surface_conflict_409"] = cross.status_code == 409


# ---------------------------------------------------------------------------
# List boundary — unstored calls never surface (except the conv store)
# ---------------------------------------------------------------------------


def _probe_list_boundary(results: dict[str, bool]) -> None:
    """Unstored calls are absent from ``GET /v1/chat/completions`` under
    every order, model filter, metadata filter and page window. The one
    list a ``store:false`` call DOES reach is a conversation's item
    list — the conv is its own store. ``/v1/responses`` and
    ``/v1/messages`` have no list route: the ``/v1`` catch-all answers
    404 in each surface's own error grammar."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    off = _chat(client, key, "list-off", store=False, metadata={"probe": "list"})
    off_id = str(off.json()["id"])
    on = _chat(client, key, "list-on", store=True, metadata={"probe": "list"})
    on_id = str(on.json()["id"])

    desc = client.get(_CHAT_PATH, headers=_h(key), params={"limit": 100, "order": "desc"})
    results["chat_list_desc_excludes_store_false"] = off_id not in _list_ids(desc)
    by_model = client.get(_CHAT_PATH, headers=_h(key), params={"model": str(on.json()["model"])})
    results["chat_list_model_filter_excludes"] = off_id not in _list_ids(
        by_model
    ) and on_id in _list_ids(by_model)
    by_meta = client.get(_CHAT_PATH, headers=_h(key), params={"metadata[probe]": "list"})
    results["chat_list_metadata_filter_excludes"] = off_id not in _list_ids(
        by_meta
    ) and on_id in _list_ids(by_meta)
    seen: set[str] = set()
    after = None
    for _ in range(64):
        page = client.get(
            _CHAT_PATH,
            headers=_h(key),
            params={"limit": 1, **({"after": after} if after else {})},
        )
        ids = _list_ids(page)
        seen |= ids
        body = page.json()
        if not body.get("has_more") or not ids:
            break
        after = str(body.get("last_id") or next(iter(ids)))
    results["chat_list_paged_window_excludes"] = off_id not in seen

    conv = _conv(client, key)
    conv_id = str(conv["id"])
    _respond(client, key, "conv-member", store=False, conversation=conv_id)
    results["conv_items_include_store_false_turn"] = (
        len(_conv_items(client, key, conv_id, limit=100)["data"]) >= 2
    )
    resp_list = client.get(_RESP_PATH, headers=_h(key))
    results["responses_list_route_404_enveloped"] = (
        resp_list.status_code == 404 and _err(resp_list).get("code") == "not_found"
    )
    msg_list = client.get(_MSG_PATH, headers=_ah(key))
    results["messages_list_route_404_anthropic_envelope"] = (
        msg_list.status_code == 404
        and msg_list.json().get("error", {}).get("type") == "not_found_error"
    )


# ---------------------------------------------------------------------------
# Subresource boundary — no orphaned items outlive a skipped envelope
# ---------------------------------------------------------------------------


def _probe_subresources(results: dict[str, bool]) -> None:
    """Subresource items share their envelope's lifetime: a ``store:
    false`` id leaves no ``/messages`` or ``/input_items`` behind, a
    DELETE drops the stored request's items with the envelope, and an
    unstored chained child leaves the stored parent's transcript
    untouched."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    off = _chat(client, key, "sub-off", store=False)
    off_id = str(off.json()["id"])
    results["store_false_id_leaves_no_subresource"] = (
        client.get(f"{_CHAT_PATH}/{off_id}/messages", headers=_h(key)).status_code == 404
    )

    on = _chat(client, key, "sub-on", store=True)
    on_id = str(on.json()["id"])
    assert client.delete(f"{_CHAT_PATH}/{on_id}", headers=_h(key)).status_code == 200
    results["chat_delete_drops_messages_subitems"] = (
        client.get(f"{_CHAT_PATH}/{on_id}/messages", headers=_h(key)).status_code == 404
    )

    parent = _respond(client, key, "parent", store=True)
    pid = str(parent.json()["id"])
    child = _respond(client, key, "child", store=False, previous_response_id=pid)
    cid = str(child.json()["id"])
    results["responses_unstored_child_items_404"] = (
        client.get(f"{_RESP_PATH}/{cid}/input_items", headers=_h(key)).status_code == 404
    )
    parent_items = client.get(f"{_RESP_PATH}/{pid}/input_items", headers=_h(key))
    results["responses_parent_items_survive_unstored_child"] = (
        parent_items.status_code == 200
        and len(parent_items.json().get("data", [])) >= 1
        and client.get(f"{_RESP_PATH}/{pid}", headers=_h(key)).status_code == 200
    )


# ---------------------------------------------------------------------------
# Restart durability — what a --state-dir restart keeps and drops
# ---------------------------------------------------------------------------


def _probe_restart(results: dict[str, bool]) -> None:
    """The retrieval index is a documented in-memory fetch cache: a
    ``--state-dir`` restart drops even ``store:true`` chat/response
    envelopes (404 on the same id), while the journaled stores survive —
    conversations (items a ``store:false`` turn appended included), the
    idempotency ledgers (a keyed unstored call still replays), managed
    key meters, eval runs and batch records. The completion log is a
    process-local ring and empties on restart."""
    state = _temporary_directory() / "state"
    c1, _api1 = _client(state_dir=state)
    key, key_id = _mint(c1, name="alice", scopes=["read", "write"])

    chat_on = _chat(c1, key, "dur-chat", store=True)
    chat_on_id = str(chat_on.json()["id"])
    chat_off = _chat(c1, key, "dur-chat-off", store=False)
    chat_off_id = str(chat_off.json()["id"])
    resp_on = _respond(c1, key, "dur-resp", store=True)
    resp_on_id = str(resp_on.json()["id"])
    conv = _conv(c1, key)
    conv_id = str(conv["id"])
    _respond(c1, key, "dur-conv-turn", store=False, conversation=conv_id)
    conv_n = len(_conv_items(c1, key, conv_id, limit=100)["data"])

    idem_body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "dur-idem"}],
        "store": False,
    }
    idem_headers = {**_h(key), "Idempotency-Key": "Idem-Dur-1"}
    idem_first = c1.post(_CHAT_PATH, json=idem_body, headers=idem_headers)
    assert idem_first.status_code == 200, idem_first.text
    idem_first_id = str(idem_first.json()["id"])

    eid = _eval_spec(c1, key, "dur-spec")
    run = c1.post(f"{_EVAL_PATH}/{eid}/runs", json={"model": "byok"}, headers=_h(key))
    run_id = str(run.json()["id"])
    done = _wait_eval_run(c1, key, eid, run_id)
    assert done.get("status") in ("completed", "failed"), done

    uses_before = _uses(c1, key_id)
    ledger_cid = str(chat_on.headers.get("x-fx1-completion-id"))
    c1.close()

    c2, _api2 = _client(state_dir=state)
    # uses bills every authenticated call — read the journaled meter
    # before this client's own probes spend anything.
    uses_at_boot = _uses(c2, key_id)
    results["restart_index_drops_stored_chat"] = (
        c2.get(f"{_CHAT_PATH}/{chat_on_id}", headers=_h(key)).status_code == 404
    )
    results["restart_index_drops_stored_response"] = (
        c2.get(f"{_RESP_PATH}/{resp_on_id}", headers=_h(key)).status_code == 404
    )
    results["restart_store_false_still_404"] = (
        c2.get(f"{_CHAT_PATH}/{chat_off_id}", headers=_h(key)).status_code == 404
    )
    results["restart_conversations_survive"] = (
        c2.get(f"{_CONV_PATH}/{conv_id}", headers=_h(key)).status_code == 200
    )
    conv_items_after = _conv_items(c2, key, conv_id, limit=100)["data"]
    results["restart_conv_items_include_store_false_turn"] = len(conv_items_after) == conv_n
    replayed = c2.post(_CHAT_PATH, json=idem_body, headers=idem_headers)
    results["restart_idem_replays_store_false"] = (
        replayed.status_code == 200
        and replayed.headers.get(_REPLAY_H) == "true"
        and str(replayed.json()["id"]) == idem_first_id
    )
    results["restart_eval_run_survives"] = (
        c2.get(f"{_EVAL_PATH}/{eid}/runs/{run_id}", headers=_h(key)).status_code == 200
    )
    results["restart_key_meters_survive"] = uses_at_boot == uses_before
    results["restart_completion_log_process_local"] = (
        c2.get(f"{_LEDGER_PATH}/{ledger_cid}", headers=_h(_ROOT)).status_code == 404
    )


# ---------------------------------------------------------------------------
# Batch per-line store — the flag is honored inside input lines
# ---------------------------------------------------------------------------


def _probe_batch_lines(results: dict[str, bool]) -> None:
    """``store`` inside a ``/v1/batches`` input line is honored per
    line: the result row still lands in the output file (the flag skips
    the retrieval index, not the batch artifact), a ``store:false``
    line's minted id answers 404, and a default-stored sibling line
    retrieves normally — on both the chat and responses endpoints."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    chat_lines = _batch_line(
        "on",
        _CHAT_PATH,
        {"model": _MODEL, "messages": [{"role": "user", "content": "b-on"}]},
    ) + _batch_line(
        "off",
        _CHAT_PATH,
        {"model": _MODEL, "messages": [{"role": "user", "content": "b-off"}], "store": False},
    )
    fid = _upload_file(client, key, "chat-lines.jsonl", chat_lines, "batch")
    b = client.post(
        _BATCHES_PATH,
        json={
            "input_file_id": fid,
            "endpoint": _CHAT_PATH,
            "completion_window": "24h",
        },
        headers=_h(key),
    )
    assert b.status_code == 200, b.text
    done = _wait_batch(client, key, str(b.json()["id"]))
    results["batch_completes"] = done.get("status") == "completed"
    out_fid = str(done.get("output_file_id"))
    rows = client.get(f"{_FILES_PATH}/{out_fid}/content", headers=_h(key))
    parsed = {
        str(json.loads(ln)["custom_id"]): json.loads(ln)
        for ln in rows.text.splitlines()
        if ln.strip()
    }
    on_line = parsed.get("on") or {}
    off_line = parsed.get("off") or {}
    on_env = (on_line.get("response") or {}).get("body") or {}
    off_env = (off_line.get("response") or {}).get("body") or {}
    results["batch_line_store_false_output_row"] = (
        str(off_env.get("object")) == "chat.completion" and str(off_env.get("id")) != ""
    )
    results["batch_chat_line_store_false_get_404"] = (
        client.get(f"{_CHAT_PATH}/{off_env.get('id')}", headers=_h(key)).status_code == 404
    )
    results["batch_chat_line_stored_get_200"] = (
        client.get(f"{_CHAT_PATH}/{on_env.get('id')}", headers=_h(key)).status_code == 200
    )

    resp_lines = _batch_line(
        "ron",
        _RESP_PATH,
        {"model": _MODEL, "input": "r-on"},
    ) + _batch_line(
        "roff",
        _RESP_PATH,
        {"model": _MODEL, "input": "r-off", "store": False},
    )
    rfid = _upload_file(client, key, "resp-lines.jsonl", resp_lines, "batch")
    rb = client.post(
        _BATCHES_PATH,
        json={
            "input_file_id": rfid,
            "endpoint": _RESP_PATH,
            "completion_window": "24h",
        },
        headers=_h(key),
    )
    assert rb.status_code == 200, rb.text
    rdone = _wait_batch(client, key, str(rb.json()["id"]))
    results["batch_responses_completes"] = rdone.get("status") == "completed"
    rrows = client.get(f"{_FILES_PATH}/{rdone.get('output_file_id')}/content", headers=_h(key))
    rparsed = {
        str(json.loads(ln)["custom_id"]): json.loads(ln)
        for ln in rrows.text.splitlines()
        if ln.strip()
    }
    ron_env = ((rparsed.get("ron") or {}).get("response") or {}).get("body") or {}
    roff_env = ((rparsed.get("roff") or {}).get("response") or {}).get("body") or {}
    results["batch_responses_line_store_false_get_404"] = (
        client.get(f"{_RESP_PATH}/{roff_env.get('id')}", headers=_h(key)).status_code == 404
    )
    results["batch_responses_line_stored_get_200"] = (
        client.get(f"{_RESP_PATH}/{ron_env.get('id')}", headers=_h(key)).status_code == 200
    )


# ---------------------------------------------------------------------------
# Eval records — always stored, no knob
# ---------------------------------------------------------------------------


def _probe_eval_records(results: dict[str, bool]) -> None:
    """Eval specs and runs live in the journaled eval store — there is
    no ``store`` knob to flip: the spec and its terminal run are always
    retrievable (restart durability pinned in the restart group)."""
    client, _api = _client()
    key, _id = _mint(client, name="alice", scopes=["read", "write"])

    eid = _eval_spec(client, key, "eval-spec")
    results["eval_spec_retrievable"] = (
        client.get(f"{_EVAL_PATH}/{eid}", headers=_h(key)).status_code == 200
    )
    run = client.post(f"{_EVAL_PATH}/{eid}/runs", json={"model": "byok"}, headers=_h(key))
    run_id = str(run.json()["id"])
    done = _wait_eval_run(client, key, eid, run_id)
    results["eval_run_retrievable_terminal"] = client.get(
        f"{_EVAL_PATH}/{eid}/runs/{run_id}", headers=_h(key)
    ).status_code == 200 and done.get("status") in ("completed", "failed")
    items = client.get(f"{_EVAL_PATH}/{eid}/runs/{run_id}/output_items", headers=_h(key))
    results["eval_run_items_persist"] = items.status_code == 200


# ---------------------------------------------------------------------------
# Envelope + metering — refusals enveloped, unstored calls billed
# ---------------------------------------------------------------------------


def _probe_envelope_metering(results: dict[str, bool]) -> None:
    """Every refusal lands in the ``{error:{...}}`` envelope, and
    ``store:false`` never discounts metering: per-key ``uses`` and
    ``tokens_used`` advance exactly like a stored call, and the
    completion-log row carries the acting credential's fingerprint."""
    client, _api = _client()
    key, key_id = _mint(client, name="alice", scopes=["read", "write"])

    missing = client.get(f"{_CHAT_PATH}/chatcmpl-{'f' * 32}", headers=_h(key))
    results["envelope_404_chat"] = (
        missing.status_code == 404 and _err(missing).get("code") == "not_found"
    )
    refused_bg = _respond(client, key, "env-bg", background=True, store=False)
    results["envelope_400_background_requires_store"] = (
        refused_bg.status_code == 400
        and _err(refused_bg).get("code") == "background_requires_store"
    )
    refused_prev = _respond(client, key, "env-prev", previous_response_id="resp_" + "0" * 32)
    results["envelope_400_previous_response_not_found"] = (
        refused_prev.status_code == 400
        and _err(refused_prev).get("code") == "previous_response_not_found"
    )

    uses0, tokens0 = _uses(client, key_id), _tokens(client, key_id)
    off = _chat(client, key, "meter-off", store=False)
    off_cid = str(off.headers.get("x-fx1-completion-id"))
    results["metering_store_false_bills_uses"] = _uses(client, key_id) == uses0 + 1
    results["metering_store_false_bills_tokens"] = _tokens(client, key_id) >= tokens0 + 5
    row = _ledger_row(client, off_cid)
    results["metering_store_false_row_key_attributed"] = (
        row is not None and row.get("key_id") == key_id
    )
    uses1, tokens1 = _uses(client, key_id), _tokens(client, key_id)
    _chat(client, key, "meter-on", store=True)
    results["metering_parity_store_true"] = (
        _uses(client, key_id) == uses1 + 1 and _tokens(client, key_id) >= tokens1 + 5
    )


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def store_audit() -> dict[str, bool]:
    """Every probe, measured end-to-end against a live app."""
    results: dict[str, bool] = {}
    with _audit_context():
        _probe_chat_boundary(results)
        _probe_responses_boundary(results)
        _probe_messages_inert(results)
        _probe_legacy_inert(results)
        _probe_streaming(results)
        _probe_background(results)
        _probe_idem(results)
        _probe_list_boundary(results)
        _probe_subresources(results)
        _probe_restart(results)
        _probe_batch_lines(results)
        _probe_eval_records(results)
        _probe_envelope_metering(results)
    return results


def store_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = store_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "store_audit",
        "schema": "store_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient (in-process ASGI) with stub backends",
            "not_verified": [
                "multi-process writers on one state_dir (single-process lock)",
                "proxy/CDN header rewriting before the app sees it",
                "real upstream provider persistence (stub backends only)",
                "the store_max eviction boundary (bounded-LRU eviction of "
                "stored envelopes is the same index, exercised elsewhere)",
                "cross-surface idempotency between OpenAI and Anthropic "
                "ledgers (separate stores by design; the shared "
                "chat/responses ledger conflict is pinned)",
            ],
        },
        "interpretation": (
            "store= is a retrieval-index knob, never an evidence knob: on "
            "chat completions and responses a store=false call executes, "
            "bills the acting credential, lands in the completion log and "
            "mints a real id that simply never resolves — GET, the "
            "messages/input_items subresources, the list surface, update "
            "and delete all answer the same 404, streaming completes its "
            "terminal frame, and background+store=false refuses fast "
            "because a background result is reachable only through the "
            "index. previous_response_id off an unstored parent fails "
            "closed 400, while a conversation keeps the turn anyway — the "
            "conv is its own store. /v1/messages and /v1/completions have "
            "no store knob at all (forced store=False at translation; an "
            "explicit store:true extra is tolerated but inert), and "
            "Idempotency-Key replay is a separate journaled ledger that "
            "replays store:false calls without repinning them. A "
            "state-dir restart drops the retrieval index even for "
            "store:true — it is a documented in-memory fetch cache — "
            "while conversations (unstored-turn items included), idem "
            "ledgers, key meters, eval runs and batches survive. Batch "
            "input lines honor store per line. No defects found."
            if ok
            else f"STORE AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(store_audit_bench(), indent=1))
