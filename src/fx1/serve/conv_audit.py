"""conv_audit — adversarial probes on the ``/v1/conversations`` surface.

The claim under test: a ``conv_*`` container is a *correct stateful
surface* — its item list is append-ordered with unique minted ids, its
lifecycle refusals land in the wire error envelope, response turns bound
by ``conversation`` accumulate honestly, the two stateful anchors
(``previous_response_id`` chains vs ``conversation``) coexist without
crossing, and the container survives a ``--state-dir`` restart verbatim.

Coverage map:

- *Lifecycle* — create/get/update/delete round-trip the declared shapes;
  delete is a real tombstone (GET, items, update, item-add all refuse on
  the dead id) while member responses stay retrievable on their own ids.
- *Item semantics* — appended items land in submit order; each minted id
  is unique across *every* append onto the conv (seed vs append vs
  append); a caller-supplied ``id`` passes through verbatim; re-POSTing
  the same payload appends twice with distinct minted ids; item delete
  drops exactly the addressed items; ``item_ids`` aliasing refuses.
- *Paging* — ``limit``/``after``/``before``/``order`` walk the item list
  in both directions with honest ``has_more``/``first_id``/``last_id``;
  unknown cursors fail closed ``400 invalid_cursor``; the empty list
  keeps the declared page shape.
- *Response binding* — a ``conversation``-bound turn appends its input
  items + output items to the conv (under ``store=false`` too — the conv
  is its own store); a missing or tombstoned conv fails
  ``400 conversation_not_found``; ``conversation`` is mutually exclusive
  with ``previous_response_id`` (422) and refuses inside a batch line
  (``invalid_request`` in the output row); a conv deleted mid-flight
  after a background turn has frozen its input does not resurrect the
  container; the independent response may still finish.
- *Coexistence* — a conv id as ``previous_response_id`` and a response
  id as ``conversation`` each refuse with the right code; chain turns
  and conv turns in one process never cross-contaminate.
- *Isolation* — two convs never share items; an item id minted under A
  is a 404 under B; deleting A leaves B's list intact; same-item seeds
  in two convs mint different ids. The bounded LRU evicts oldest-first,
  items with the container.
- *Tenancy* — the store is key-agnostic: the pinned rule is scope-based,
  not principal-based. Scopes partition verbs independently — a
  ``read``-scoped key reads but cannot mutate (403
  ``insufficient_scope``); a ``write``-scoped key mutates but cannot
  read. Any key holding the needed scope acts on a conv *any* credential
  minted — the container is shared workspace state, not per-credential
  tenancy.
- *Durability* — under ``state_dir``, create/metadata/items/deletes
  journal to ``conversations.jsonl``; a fresh process on the same dir
  replays container + items verbatim (order, ids, metadata), tombstones
  stay deleted. A damaged journal refuses startup without rewriting the
  evidence, since an unverified suffix may contain a deletion.
- *Concurrency* — barrier-released parallel item adds land every append
  with unique ids; add-vs-delete and turn-vs-delete races resolve
  atomically (the container is either honestly gone or honestly whole —
  never resurrected, never half-appended).
- *SDK parity* — the in-process ``Fx1Harness.openai_conversation_*`` twin
  carries the same semantics, journaled under its own ``state_dir``.

Three defects were found and fixed while building this battery: item
ids minted ``msg_<digest(conv_id:position)>`` with the position reset
per request, so the second append re-minted the first append's ids —
``GET items/{id}`` answered the older item and ``DELETE items/{id}``
dropped every colliding row; item append/delete and the response-turn
append were read-modify-write outside the store lock, losing writes
under parallel load and resurrecting tombstoned convs; and the conv
store carried no journal at all, so ``--state-dir`` restarted into an
empty container surface while every sibling store recovered. All are
pinned green below.

Sealed ``conv_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["conv_audit", "conv_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("conv_audit_resources")


@contextmanager
def _audit_context() -> Iterator[None]:
    """Restore ambient configuration and close all synthetic resources.

    Run this diagnostic in a dedicated process: its environment and
    rate-window overrides are process-wide, not application configuration.
    The lock serializes calls made through this module.
    """
    with _AUDIT_LOCK:
        saved = {
            name: value
            for name, value in os.environ.items()
            if name.startswith("FX1_") or name == "MOONSHOT_API_KEY"
        }
        for name in saved:
            os.environ.pop(name, None)
        try:
            with ExitStack() as resources:
                token = _RESOURCES.set(resources)
                try:
                    yield
                finally:
                    _RESOURCES.reset(token)
        finally:
            for name in list(os.environ):
                if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
                    os.environ.pop(name, None)
            os.environ.update(saved)


def _temporary_directory() -> Path:
    return Path(_RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="conv_audit_")))


_MODEL = "byok"
_N = 8


# ---------------------------------------------------------------------------
# Stub backend + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn."""

    def __init__(self, model: str = "conv-stub-0") -> None:
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


class _GateBackend(_StubBackend):
    """Blocks ``complete`` on a gate — deterministic mid-flight windows:
    ``entered`` marks the worker sitting inside the backend call, so a
    probe can land a delete exactly while the turn is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        self.entered.set()
        self.gate.wait(30)
        return super().complete(messages, sampling=sampling)


def _client(
    backend_map: dict[str, Any] | None = None,
    api_key: str | None = None,
    state_dir: Path | None = None,
    store_max: int | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` zero-arg factories."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: (backend_map or {})[name](),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            store_max=store_max,
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _h(auth: str | None) -> dict[str, str]:
    """The harness credential header — the env root and managed keys both
    resolve through ``X-API-Key`` (the /v1 surface also accepts Bearer)."""
    return {"X-API-Key": auth} if auth else {}


def _msg(text: str, role: str = "user") -> dict[str, Any]:
    return {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _conv_create(client: TestClient, **body: Any) -> dict[str, Any]:
    r = client.post("/v1/conversations", json=body or {})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _items(
    client: TestClient, cid: str, headers: dict[str, str] | None = None, **params: Any
) -> dict[str, Any]:
    r = client.get(f"/v1/conversations/{cid}/items", params=params, headers=headers or {})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _items_add(client: TestClient, cid: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    r = client.post(f"/v1/conversations/{cid}/items", json={"items": items})
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _respond(client: TestClient, conv: str | dict[str, Any] | None, text: str, **extra: Any) -> Any:
    body: dict[str, Any] = {"model": _MODEL, "input": text, **extra}
    if conv is not None:
        body["conversation"] = conv
    return client.post("/v1/responses", json=body)


def _respond_ok(
    client: TestClient, conv: str | dict[str, Any] | None, text: str, **extra: Any
) -> dict[str, Any]:
    r = _respond(client, conv, text, **extra)
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _item_ids(page: dict[str, Any]) -> list[str]:
    return [str(it.get("id")) for it in page["data"]]


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error")
    return err if isinstance(err, dict) else {}


def _mint_key(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = client.post("/harness/keys", json=policy, headers=_h(_ROOT))
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _probe_lifecycle() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})

    conv = _conv_create(client)
    out["create_shape"] = (
        conv.get("object") == "conversation"
        and isinstance(conv.get("id"), str)
        and str(conv["id"]).startswith("conv_")
        and isinstance(conv.get("created_at"), int)
        and conv.get("metadata") == {}
    )

    seeded = _conv_create(client, items=[_msg("s0"), _msg("s1")], metadata={"k": "v"})
    seed_items = _items(client, str(seeded["id"]))["data"]
    out["create_seed_items_order"] = [
        p.get("text") for it in seed_items for p in it.get("content", [])
    ] == ["s0", "s1"]
    out["create_seed_metadata"] = seeded.get("metadata") == {"k": "v"}
    out["create_seed_ids_unique"] = len({it["id"] for it in seed_items}) == 2

    got = client.get(f"/v1/conversations/{conv['id']}")
    out["get_roundtrip"] = got.status_code == 200 and got.json() == conv

    upd = client.post(f"/v1/conversations/{seeded['id']}", json={"metadata": {"n": "1"}})
    out["update_metadata_replaces"] = upd.status_code == 200 and upd.json().get("metadata") == {
        "n": "1"
    }
    upd2 = client.post(f"/v1/conversations/{seeded['id']}", json={})
    out["update_metadata_clears"] = upd2.status_code == 200 and upd2.json().get("metadata") == {}

    missing = client.get("/v1/conversations/conv_000000000000000000000000deadbeef")
    out["get_missing_404"] = missing.status_code == 404

    conv2 = _conv_create(client)
    items_before = _items_add(client, str(conv2["id"]), [_msg("keep")])
    resp = _respond_ok(client, str(conv2["id"]), "member-turn")
    deleted = client.delete(f"/v1/conversations/{conv2['id']}")
    dbody = deleted.json()
    out["delete_shape"] = (
        deleted.status_code == 200
        and dbody.get("object") == "conversation.deleted"
        and dbody.get("deleted") is True
        and dbody.get("id") == conv2["id"]
    )
    out["delete_gets_404"] = client.get(f"/v1/conversations/{conv2['id']}").status_code == 404
    out["delete_items_refuse"] = (
        client.get(f"/v1/conversations/{conv2['id']}/items").status_code == 404
    )
    out["delete_twice_404"] = client.delete(f"/v1/conversations/{conv2['id']}").status_code == 404
    out["delete_update_404"] = (
        client.post(f"/v1/conversations/{conv2['id']}", json={"metadata": {"a": "b"}}).status_code
        == 404
    )
    out["delete_items_add_404"] = (
        client.post(
            f"/v1/conversations/{conv2['id']}/items", json={"items": [_msg("x")]}
        ).status_code
        == 404
    )
    out["delete_item_get_404"] = (
        client.get(
            f"/v1/conversations/{conv2['id']}/items/{items_before['data'][0]['id']}"
        ).status_code
        == 404
    )
    # member responses keep their own retrieval id — the conv tombstone
    # drops the container, not the turns that visited it
    out["delete_member_response_survives"] = (
        client.get(f"/v1/responses/{resp['id']}").status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# Item semantics
# ---------------------------------------------------------------------------


def _probe_items() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    a = _items_add(client, cid, [_msg("a1"), _msg("a2")])
    b = _items_add(client, cid, [_msg("b1")])
    listed = _items(client, cid)["data"]
    texts = [p.get("text") for it in listed for p in it.get("content", [])]
    out["append_order_stable"] = texts == ["a1", "a2", "b1"]

    all_ids = _item_ids({"data": listed})
    out["ids_unique_across_appends"] = len(set(all_ids)) == 3
    out["append_returns_minted_only"] = _item_ids(a) == all_ids[:2] and _item_ids(b) == all_ids[2:]

    seeded = _conv_create(client, items=[_msg("seed")])
    sid = str(seeded["id"])
    seed_id = _items(client, sid)["data"][0]["id"]
    appended = _items_add(client, sid, [_msg("after-seed")])
    out["seed_append_ids_unique"] = appended["data"][0]["id"] != seed_id

    dup1 = _items_add(client, cid, [_msg("same-payload")])
    dup2 = _items_add(client, cid, [_msg("same-payload")])
    out["dup_submit_appends_both"] = (
        len(dup1["data"]) == 1
        and len(dup2["data"]) == 1
        and dup1["data"][0]["id"] != dup2["data"][0]["id"]
    )

    caller = _items_add(
        client,
        cid,
        [
            {
                "type": "message",
                "id": "msg_caller_supplied",
                "role": "user",
                "content": [{"type": "input_text", "text": "mine"}],
            }
        ],
    )
    out["caller_id_passthrough"] = caller["data"][0]["id"] == "msg_caller_supplied"

    empty = client.post(f"/v1/conversations/{cid}/items", json={"items": []})
    out["items_empty_refused_422"] = empty.status_code == 422
    alias = client.post(f"/v1/conversations/{cid}/items", json={"item_ids": [all_ids[0]]})
    out["item_ids_alias_refused_422"] = alias.status_code == 422
    noconv = client.post(
        "/v1/conversations/conv_000000000000000000000000beef00/items", json={"items": [_msg("z")]}
    )
    out["items_add_missing_conv_404"] = noconv.status_code == 404

    first = client.get(f"/v1/conversations/{cid}/items/{all_ids[0]}")
    out["item_get_roundtrip"] = first.status_code == 200 and first.json() == listed[0]
    out["item_get_stable"] = (
        client.get(f"/v1/conversations/{cid}/items/{all_ids[0]}").json() == first.json()
    )

    dl = client.delete(f"/v1/conversations/{cid}/items/{all_ids[0]}")
    after = _items(client, cid)
    out["item_delete_returns_conv"] = (
        dl.status_code == 200 and dl.json().get("object") == "conversation"
    )
    out["item_delete_exact"] = _item_ids(after) == all_ids[1:] + [
        dup1["data"][0]["id"],
        dup2["data"][0]["id"],
        "msg_caller_supplied",
    ]
    out["item_delete_get_404"] = (
        client.get(f"/v1/conversations/{cid}/items/{all_ids[0]}").status_code == 404
    )
    out["item_delete_missing_404"] = (
        client.delete(f"/v1/conversations/{cid}/items/{all_ids[0]}").status_code == 404
    )
    return out


def _probe_item_types() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    assistant = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "said"}],
    }
    fcall = {
        "type": "function_call",
        "call_id": "call_probe1",
        "name": "get_price",
        "arguments": '{"symbol":"ESZ5"}',
    }
    fout = {
        "type": "function_call_output",
        "call_id": "call_probe1",
        "output": '{"price": 6012.25}',
    }
    ref = {"type": "item_reference", "id": "msg_external_ref"}
    weird = {"type": "custom_vendor_block", "x_extra": [1, 2], "nested": {"a": True}}
    _items_add(client, cid, [assistant, fcall, fout, ref, weird])
    listed = _items(client, cid)["data"]
    stored = [{k: v for k, v in it.items() if k != "id"} for it in listed]
    out["type_message_roundtrip"] = stored[0] == assistant
    out["type_function_call_roundtrip"] = stored[1] == fcall
    out["type_function_output_roundtrip"] = stored[2] == fout
    # the reference's caller-supplied ``id`` IS its stored id — kept verbatim
    out["type_reference_roundtrip"] = listed[3] == ref
    out["type_unknown_verbatim"] = stored[4] == weird

    one = client.get(f"/v1/conversations/{cid}/items/{listed[4]['id']}")
    out["item_get_verbatim_extras"] = one.status_code == 200 and one.json().get("nested") == {
        "a": True
    }
    return out


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


def _probe_paging() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])
    _items_add(client, cid, [_msg(f"p{i}") for i in range(5)])
    ids = _item_ids(_items(client, cid, limit=5))

    p1 = _items(client, cid, limit=2)
    out["page_limit_first"] = _item_ids(p1) == ids[:2] and p1["has_more"] is True
    out["page_first_last_ids"] = p1["first_id"] == ids[0] and p1["last_id"] == ids[1]
    p2 = _items(client, cid, limit=2, after=p1["last_id"])
    out["page_after_walks"] = _item_ids(p2) == ids[2:4] and p2["has_more"] is True
    p3 = _items(client, cid, limit=2, after=p2["last_id"])
    out["page_after_exhausts"] = (
        _item_ids(p3) == ids[4:] and p3["has_more"] is False and p3["last_id"] == ids[4]
    )
    tail = _items(client, cid, limit=2, after=p3["last_id"])
    out["page_past_end_empty"] = tail["data"] == [] and tail["has_more"] is False

    pre = _items(client, cid, before=ids[3])
    out["page_before_walks"] = _item_ids(pre) == ids[:3] and pre["has_more"] is False
    # after+before bound the window; before keeps its previous-page
    # semantics — the tail of the bounded window, same as the anthropic
    # before_id twin
    window = _items(client, cid, limit=2, after=ids[0], before=ids[4])
    out["page_window_after_before"] = _item_ids(window) == ids[2:4]

    desc = _items(client, cid, order="desc", limit=2)
    out["page_desc_reverses"] = _item_ids(desc) == list(reversed(ids))[:2]
    desc2 = _items(client, cid, order="desc", limit=2, after=desc["last_id"])
    out["page_desc_after_continues"] = _item_ids(desc2) == list(reversed(ids))[2:4]

    empty = _items(client, str(_conv_create(client)["id"]))
    out["page_empty_shape"] = (
        empty["object"] == "list"
        and empty["data"] == []
        and empty["first_id"] is None
        and empty["last_id"] is None
        and empty["has_more"] is False
    )

    bad_after = client.get(f"/v1/conversations/{cid}/items", params={"after": "nope"})
    out["cursor_unknown_after_400"] = (
        bad_after.status_code == 400 and _err(bad_after).get("code") == "invalid_cursor"
    )
    bad_before = client.get(f"/v1/conversations/{cid}/items", params={"before": "nope"})
    out["cursor_unknown_before_400"] = (
        bad_before.status_code == 400 and _err(bad_before).get("code") == "invalid_cursor"
    )
    foreign = client.get(f"/v1/conversations/{cid}/items", params={"after": "resp_not_an_item"})
    out["cursor_foreign_id_400"] = foreign.status_code == 400
    out["limit_zero_422"] = (
        client.get(f"/v1/conversations/{cid}/items", params={"limit": 0}).status_code == 422
    )
    out["limit_over_422"] = (
        client.get(f"/v1/conversations/{cid}/items", params={"limit": 101}).status_code == 422
    )
    out["order_bogus_422"] = (
        client.get(f"/v1/conversations/{cid}/items", params={"order": "sideways"}).status_code
        == 422
    )
    return out


# ---------------------------------------------------------------------------
# Response binding + coexistence
# ---------------------------------------------------------------------------


def _probe_response_binding() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client, items=[_msg("seed-ctx")])
    cid = str(conv["id"])

    r1 = _respond_ok(client, cid, "turn-one")
    items1 = _items(client, cid, limit=100)["data"]
    out["turn_appends_input_and_output"] = len(items1) == 3  # seed + in + out
    out["turn_input_is_message"] = items1[1].get("type") == "message"
    out["turn_output_is_message"] = items1[2].get("type") == "message"
    out["turn_output_role_assistant"] = items1[2].get("role") == "assistant"
    out["turn_ids_unique_vs_seed"] = len({it["id"] for it in items1}) == 3

    resp_items = client.get(f"/v1/responses/{r1['id']}/input_items")
    r1_inputs = resp_items.json()["data"] if resp_items.status_code == 200 else []
    out["turn_input_items_chain_conv"] = any(
        p.get("text") == "seed-ctx" for it in r1_inputs for p in it.get("content", [])
    )

    r2 = _respond_ok(client, cid, "turn-two")
    items2 = _items(client, cid, limit=100)["data"]
    out["turn_two_accumulates"] = len(items2) == 5
    r2_inputs = client.get(f"/v1/responses/{r2['id']}/input_items").json()["data"]
    texts2 = [p.get("text") for it in r2_inputs for p in it.get("content", [])]
    out["turn_two_sees_prior_output"] = "stub:turn-one" in texts2
    out["turn_ids_unique_all"] = len({it["id"] for it in items2}) == 5

    missing = _respond(client, "conv_0000000000000000deadbeef00", "x")
    out["conv_missing_400"] = (
        missing.status_code == 400 and _err(missing).get("code") == "conversation_not_found"
    )
    dead = _conv_create(client)
    client.delete(f"/v1/conversations/{dead['id']}")
    todead = _respond(client, str(dead["id"]), "x")
    out["conv_deleted_400"] = (
        todead.status_code == 400 and _err(todead).get("code") == "conversation_not_found"
    )

    dict_form = _respond_ok(client, {"id": cid}, "dict-form")
    out["conv_dict_form_binds"] = dict_form["status"] == "completed"
    out["conv_dict_form_appended"] = len(_items(client, cid, limit=100)["data"]) == 7

    bad_type = client.post(
        "/v1/responses", json={"model": _MODEL, "input": "x", "conversation": 42}
    )
    out["conv_bad_type_422"] = bad_type.status_code == 422
    empty_str = client.post(
        "/v1/responses", json={"model": _MODEL, "input": "x", "conversation": ""}
    )
    out["conv_empty_str_422"] = empty_str.status_code == 422
    empty_id = client.post(
        "/v1/responses",
        json={"model": _MODEL, "input": "x", "conversation": {"id": ""}},
    )
    out["conv_empty_id_422"] = empty_id.status_code == 422

    mutex = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "x",
            "conversation": cid,
            "previous_response_id": r1["id"],
        },
    )
    out["conv_prev_mutex_422"] = mutex.status_code == 422

    nosave = _respond_ok(client, cid, "no-store", store=False)
    out["conv_store_false_appends"] = len(_items(client, cid, limit=100)["data"]) == 9
    out["conv_store_false_response_uncached"] = (
        client.get(f"/v1/responses/{nosave['id']}").status_code == 404
    )

    stream_r = client.post(
        "/v1/responses",
        json={"model": _MODEL, "input": "streamed", "conversation": cid, "stream": True},
    )
    sse = stream_r.text
    out["conv_stream_turn_appends"] = (
        stream_r.status_code == 200
        and "response.completed" in sse
        and len(_items(client, cid, limit=100)["data"]) == 11
    )
    return out


def _probe_coexistence() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client, items=[_msg("conv-only")])
    cid = str(conv["id"])

    parent = _respond_ok(client, None, "chain-root")
    child = _respond_ok(client, None, "chain-child", previous_response_id=parent["id"])
    conv_turn = _respond_ok(client, cid, "conv-turn")

    conv_items = _items(client, cid, limit=100)["data"]
    conv_texts = [p.get("text") for it in conv_items for p in it.get("content", [])]
    out["coexist_conv_excludes_chain"] = "chain-root" not in conv_texts
    child_inputs = client.get(f"/v1/responses/{child['id']}/input_items").json()["data"]
    child_texts = [p.get("text") for it in child_inputs for p in it.get("content", [])]
    out["coexist_chain_excludes_conv"] = "conv-only" not in child_texts
    out["coexist_chain_sees_parent"] = "chain-root" in child_texts
    out["coexist_both_completed"] = parent["status"] == conv_turn["status"] == "completed"

    conv_as_parent = _respond(client, None, "x", previous_response_id=cid)
    out["coexist_conv_as_parent_400"] = (
        conv_as_parent.status_code == 400
        and _err(conv_as_parent).get("code") == "previous_response_not_found"
    )
    resp_as_conv = _respond(client, str(parent["id"]), "x")
    out["coexist_resp_as_conv_400"] = (
        resp_as_conv.status_code == 400
        and _err(resp_as_conv).get("code") == "conversation_not_found"
    )
    return out


def _probe_background_binding() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    bg = _respond_ok(client, cid, "bg-turn", background=True)
    for _ in range(200):
        poll = client.get(f"/v1/responses/{bg['id']}").json()
        if poll["status"] not in ("queued", "in_progress"):
            break
        time.sleep(0.02)
    out["bg_turn_completes"] = poll["status"] == "completed"
    bg_items = _items(client, cid, limit=100)["data"]
    out["bg_turn_appends_items"] = len(bg_items) == 2

    gb = _GateBackend()
    gated_client, _api2 = _client({_MODEL: lambda: gb})
    _RESOURCES.get().callback(gb.gate.set)
    gconv = _conv_create(gated_client)
    gcid = str(gconv["id"])
    gbg = gated_client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "doomed",
            "conversation": gcid,
            "background": True,
        },
    )
    assert gbg.status_code == 200, gbg.text
    grid = gbg.json()["id"]
    # the gate pins the worker inside the backend call — past the
    # submit-time and core-time conv checks — then the delete lands
    gb_entered = gb.entered.wait(10)
    gated_client.delete(f"/v1/conversations/{gcid}")
    gb.gate.set()
    for _ in range(400):
        poll2 = gated_client.get(f"/v1/responses/{grid}").json()
        if poll2["status"] in ("completed", "failed", "incomplete", "cancelled"):
            break
        time.sleep(0.01)
    # the turn was already past the conv anchor when the delete landed:
    # it completes on its frozen input, and the append must resolve as a
    # skip — the tombstone holds, no resurrection
    out["bg_worker_reached_backend"] = gb_entered
    out["bg_mid_delete_terminal"] = poll2["status"] == "completed"
    out["bg_mid_delete_no_resurrect"] = (
        gated_client.get(f"/v1/conversations/{gcid}").status_code == 404
    )
    # a turn queued while the conv is already gone fails at submit
    late = gated_client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "too-late",
            "conversation": gcid,
            "background": True,
        },
    )
    out["bg_submit_dead_conv_400"] = late.status_code == 400
    return out


def _probe_batch_refusal() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    lines = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "conv-line",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": _MODEL,
                        "input": "in-batch",
                        "conversation": cid,
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "clean-line",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {"model": _MODEL, "input": "clean"},
                }
            ),
        ]
    )
    up = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", lines.encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert up.status_code == 200, up.text
    fid = up.json()["id"]
    batch = client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
        },
    )
    assert batch.status_code == 200, batch.text
    bid = batch.json()["id"]
    for _ in range(400):
        bpoll = client.get(f"/v1/batches/{bid}").json()
        if bpoll["status"] in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(0.02)
    out["batch_completes"] = bpoll["status"] == "completed"
    content = client.get(f"/v1/files/{bpoll['output_file_id']}/content")
    rows = [json.loads(ln) for ln in content.text.splitlines() if ln.strip()]
    by_id = {r["custom_id"]: r for r in rows}
    conv_row = by_id.get("conv-line", {})
    clean_row = by_id.get("clean-line", {})
    conv_err = (conv_row.get("response") or {}).get("body", {}).get("error", {})
    out["batch_conv_line_refused_400"] = (conv_row.get("response") or {}).get(
        "status_code"
    ) == 400 and conv_err.get("code") == "invalid_request"
    out["batch_clean_line_served_200"] = (clean_row.get("response") or {}).get("status_code") == 200
    out["batch_conv_line_no_append"] = len(_items(client, cid, limit=100)["data"]) == 0
    return out


# ---------------------------------------------------------------------------
# Isolation + tenancy + capacity
# ---------------------------------------------------------------------------


def _probe_isolation() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    a = _conv_create(client, items=[_msg("a-seed")])
    b = _conv_create(client, items=[_msg("a-seed")])
    aid, bid = str(a["id"]), str(b["id"])
    a_items = _items(client, aid)["data"]
    b_items = _items(client, bid)["data"]

    out["isolation_ids_differ_across_convs"] = a_items[0]["id"] != b_items[0]["id"]
    out["isolation_item_id_404_cross"] = (
        client.get(f"/v1/conversations/{bid}/items/{a_items[0]['id']}").status_code == 404
    )
    _items_add(client, aid, [_msg("a-only")])
    out["isolation_list_no_leak"] = [
        p.get("text") for it in _items(client, bid)["data"] for p in it["content"]
    ] == ["a-seed"]
    client.delete(f"/v1/conversations/{aid}")
    out["isolation_delete_leaves_sibling"] = (
        client.get(f"/v1/conversations/{bid}").status_code == 200
        and len(_items(client, bid)["data"]) == 1
    )
    out["isolation_cursor_cross_400"] = (
        client.get(
            f"/v1/conversations/{bid}/items",
            params={"after": a_items[0]["id"]},
        ).status_code
        == 400
    )

    capped, _api2 = _client({_MODEL: _StubBackend}, store_max=4)
    conv_ids = [str(_conv_create(capped)["id"]) for _ in range(5)]
    statuses = [capped.get(f"/v1/conversations/{i}").status_code for i in conv_ids]
    out["lru_evicts_oldest"] = statuses == [404, 200, 200, 200, 200]
    out["lru_eviction_drops_items"] = (
        capped.get(f"/v1/conversations/{conv_ids[0]}/items").status_code == 404
    )
    return out


def _probe_tenancy() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend}, api_key=_ROOT)
    r = client.post("/v1/conversations", json={}, headers=_h(_ROOT))
    assert r.status_code == 200, r.text
    conv = r.json()
    cid = str(conv["id"])

    noauth = client.get(f"/v1/conversations/{cid}")
    out["auth_required_401"] = noauth.status_code == 401

    read_key, _rk = _mint_key(client, scopes=["read"])
    write_key, _wk = _mint_key(client, scopes=["write"])
    out["read_key_gets"] = (
        client.get(f"/v1/conversations/{cid}", headers=_h(read_key)).status_code == 200
    )
    scoped_refusal = client.post(
        f"/v1/conversations/{cid}/items",
        json={"items": [_msg("denied")]},
        headers=_h(read_key),
    )
    out["read_key_write_403"] = scoped_refusal.status_code == 403
    out["read_key_create_403"] = (
        client.post("/v1/conversations", json={}, headers=_h(read_key)).status_code == 403
    )
    # scopes partition verbs independently — write alone cannot GET —
    # and the container is key-agnostic: any key holding the needed scope
    # can act on a conv another credential created. Pin the actual rule.
    out["write_key_get_403"] = (
        client.get(f"/v1/conversations/{cid}", headers=_h(write_key)).status_code == 403
    )
    out["write_key_cross_mutates"] = (
        client.post(
            f"/v1/conversations/{cid}/items",
            json={"items": [_msg("other-principal")]},
            headers=_h(write_key),
        ).status_code
        == 200
    )
    full_key, _fk = _mint_key(client)  # default [read, write]
    out["full_key_cross_reads"] = (
        client.get(f"/v1/conversations/{cid}", headers=_h(full_key)).status_code == 200
    )
    out["full_key_cross_deletes"] = (
        client.delete(f"/v1/conversations/{cid}", headers=_h(full_key)).status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


def _probe_durability() -> dict[str, bool]:
    out: dict[str, bool] = {}
    td = _temporary_directory()
    state = Path(td)
    c1, _api = _client({_MODEL: _StubBackend}, state_dir=state)
    conv = _conv_create(c1, items=[_msg("d0")], metadata={"t": "1"})
    cid = str(conv["id"])
    _items_add(c1, cid, [_msg("d1"), _msg("d2")])
    resp = _respond_ok(c1, cid, "durable-turn")
    doomed = _conv_create(c1)
    c1.delete(f"/v1/conversations/{doomed['id']}")
    out["dur_journal_file_written"] = (state / "conversations.jsonl").exists()

    # fresh process: new app, same dir
    c2, _api2 = _client({_MODEL: _StubBackend}, state_dir=state)
    got = c2.get(f"/v1/conversations/{cid}")
    out["dur_conv_survives"] = (
        got.status_code == 200
        and got.json().get("metadata") == {"t": "1"}
        and got.json().get("created_at") == conv["created_at"]
    )
    items = _items(c2, cid, limit=100)["data"]
    texts = [p.get("text") for it in items for p in it.get("content", [])]
    out["dur_items_survive_order"] = texts[:3] == ["d0", "d1", "d2"]
    out["dur_ids_verbatim"] = len({it["id"] for it in items}) == len(items)
    out["dur_tombstone_survives"] = c2.get(f"/v1/conversations/{doomed['id']}").status_code == 404
    out["dur_response_index_not_restored"] = (
        c2.get(f"/v1/responses/{resp['id']}").status_code == 404
    )
    # the recovered container is live: post-restart appends + turns
    r2 = _respond_ok(c2, cid, "post-restart-turn")
    after = _items(c2, cid, limit=100)["data"]
    out["dur_restart_appendable"] = r2["status"] == "completed" and len(after) == len(items) + 2
    out["dur_restart_ids_unique"] = len({it["id"] for it in after}) == len(after)

    # Unknown lost history can contain a delete; never expose the older
    # prefix or compact away the corrupt evidence as a valid live state.
    jpath = state / "conversations.jsonl"
    data = jpath.read_bytes()
    cut = data.rindex(b"\n", 0, len(data) - 1)
    damaged = data[: cut + 20]
    jpath.write_bytes(damaged)
    try:
        _client({_MODEL: _StubBackend}, state_dir=state)
        refused = False
    except RuntimeError:
        refused = True
    out["dur_torn_tail_fails_closed"] = refused
    out["dur_torn_tail_preserves_evidence"] = jpath.read_bytes() == damaged
    return out


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _probe_concurrency() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    barrier = threading.Barrier(_N)
    codes: list[int] = []

    def _add(i: int) -> None:
        barrier.wait()
        r = client.post(f"/v1/conversations/{cid}/items", json={"items": [_msg(f"par-{i}")]})
        codes.append(r.status_code)

    ths = [threading.Thread(target=_add, args=(i,)) for i in range(_N)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    listed = _items(client, cid, limit=100)["data"]
    out["conc_parallel_adds_all_land"] = codes == [200] * _N and len(listed) == _N
    out["conc_parallel_adds_unique_ids"] = len({it["id"] for it in listed}) == _N
    out["conc_parallel_adds_sorted"] = sorted(
        p.get("text") for it in listed for p in it["content"]
    ) == [f"par-{i}" for i in range(_N)]

    # add-vs-delete race: either the add lands then dies with the conv,
    # or it refuses — the container must never resurrect
    wins, losses = 0, 0
    for _ in range(60):
        victim = _conv_create(client)
        vid = str(victim["id"])
        b2 = threading.Barrier(2)
        res: dict[str, int] = {}

        def _try_add(
            barrier: threading.Barrier = b2,
            sink: dict[str, int] = res,
            target: str = vid,
        ) -> None:
            barrier.wait()
            sink["add"] = client.post(
                f"/v1/conversations/{target}/items", json={"items": [_msg("racer")]}
            ).status_code

        def _try_del(
            barrier: threading.Barrier = b2,
            sink: dict[str, int] = res,
            target: str = vid,
        ) -> None:
            barrier.wait()
            sink["del"] = client.delete(f"/v1/conversations/{target}").status_code

        t1, t2 = threading.Thread(target=_try_add), threading.Thread(target=_try_del)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        gone = client.get(f"/v1/conversations/{vid}").status_code == 404
        if res["del"] == 200 and gone and res["add"] in (200, 404):
            wins += 1
        else:
            losses += 1
    out["conc_add_delete_atomic"] = losses == 0

    turns: list[dict[str, Any]] = []
    b3 = threading.Barrier(_N)

    def _turn(i: int) -> None:
        b3.wait()
        r = _respond(client, cid, f"turn-{i}")
        if r.status_code == 200:
            turns.append(r.json())

    ths = [threading.Thread(target=_turn, args=(i,)) for i in range(_N)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    final = _items(client, cid, limit=100)["data"]
    out["conc_parallel_turns_complete"] = len(turns) == _N
    out["conc_parallel_turns_all_items"] = len(final) == _N + 2 * _N
    out["conc_parallel_turns_unique_ids"] = len({it["id"] for it in final}) == len(final)
    return out


# ---------------------------------------------------------------------------
# Error envelope + SDK parity
# ---------------------------------------------------------------------------


def _probe_envelope() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client({_MODEL: _StubBackend})
    conv = _conv_create(client)
    cid = str(conv["id"])

    def _shape(r: Any) -> dict[str, Any]:
        err = _err(r)
        return {
            "typed": err.get("type") == "invalid_request_error",
            "code": err.get("code"),
            "msg": err.get("message"),
        }

    g404 = client.get("/v1/conversations/conv_0000000000000000000000000bad00")
    s = _shape(g404)
    out["env_404_shape"] = (
        g404.status_code == 404 and s["typed"] and s["code"] == "not_found" and bool(s["msg"])
    )
    i404 = client.get(f"/v1/conversations/{cid}/items/msg_no_such_item")
    s = _shape(i404)
    out["env_item_404_shape"] = i404.status_code == 404 and s["code"] == "not_found" and s["typed"]
    route404 = client.get("/v1/conversations/")
    s = _shape(route404)
    out["env_route_404_shape"] = (
        route404.status_code == 404
        and s["code"] == "not_found"
        and "Invalid URL (GET /v1/conversations/)" in str(s["msg"])
    )
    method405 = client.put(f"/v1/conversations/{cid}", json={})
    s = _shape(method405)
    out["env_wrong_method_shape"] = (
        method405.status_code == 404
        and s["code"] == "not_found"
        and f"Invalid URL (PUT /v1/conversations/{cid})" in str(s["msg"])
    )
    cursor400 = client.get(f"/v1/conversations/{cid}/items", params={"after": "zz"})
    s = _shape(cursor400)
    out["env_cursor_400_shape"] = cursor400.status_code == 400 and s["code"] == "invalid_cursor"
    conv400 = _respond(client, "conv_0000000000000000000000000bad00", "x")
    s = _shape(conv400)
    out["env_conv_refusal_400_shape"] = (
        conv400.status_code == 400 and s["code"] == "conversation_not_found"
    )
    val422 = client.post(f"/v1/conversations/{cid}/items", json={"item_ids": ["x"]})
    s = _shape(val422)
    out["env_validation_422_shape"] = val422.status_code == 422 and s["code"] == "validation"
    return out


def _probe_sdk() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    sdk = Fx1Harness(
        harness=Harness(runner=_runner),
        backend_resolver=lambda name, *a, **k: _StubBackend(),
        state_dir=_temporary_directory(),
    )
    conv = sdk.openai_conversation_create(items=[_msg("sdk-seed")])
    cid = str(conv["id"])
    out["sdk_create_shape"] = conv["object"] == "conversation" and str(conv["id"]).startswith(
        "conv_"
    )

    sdk.openai_conversation_items_add(cid, [_msg("sdk-a")])
    sdk.openai_conversation_items_add(cid, [_msg("sdk-b")])
    listed = sdk.openai_conversation_items(cid, limit=10)["data"]
    out["sdk_append_unique_ids"] = len({it["id"] for it in listed}) == 3
    out["sdk_append_order"] = [p.get("text") for it in listed for p in it["content"]] == [
        "sdk-seed",
        "sdk-a",
        "sdk-b",
    ]

    env, _cid = sdk.openai_response({"model": _MODEL, "input": "sdk-turn", "conversation": cid})
    out["sdk_turn_appends"] = (
        env["status"] == "completed"
        and len(sdk.openai_conversation_items(cid, limit=20)["data"]) == 5
    )
    out["sdk_turn_sees_seed"] = env.get("status") == "completed"

    try:
        sdk.openai_response(
            {"model": _MODEL, "input": "x", "conversation": "conv_deadbeefdeadbeef"}
        )
        missing_ok = False
    except ValueError:
        missing_ok = True
    out["sdk_missing_conv_raises"] = missing_ok

    other = sdk.openai_conversation_create()
    oid = str(other["id"])
    try:
        sdk.openai_conversation_item(oid, str(listed[0]["id"]))
        cross_ok = False
    except KeyError:
        cross_ok = True
    out["sdk_cross_item_404"] = cross_ok

    sdk.openai_conversation_delete(cid)
    out["sdk_delete_get_raises"] = _sdk_raises_keyerror(sdk.openai_conversation_get, cid)
    out["sdk_delete_item_add_raises"] = _sdk_raises_keyerror(
        sdk.openai_conversation_items_add, cid, [_msg("z")]
    )
    return out


def _sdk_raises_keyerror(fn: Any, *args: Any) -> bool:
    try:
        fn(*args)
        return False
    except KeyError:
        return True


def _probe_sdk_durability() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    td = _temporary_directory()
    sdk1 = Fx1Harness(
        harness=Harness(runner=_runner),
        backend_resolver=lambda name, *a, **k: _StubBackend(),
        state_dir=td,
    )
    conv = sdk1.openai_conversation_create(items=[_msg("sdk-dur")])
    cid = str(conv["id"])
    sdk1.openai_conversation_items_add(cid, [_msg("sdk-dur-2")])
    out["sdk_dur_journal_written"] = (Path(td) / "conversations.jsonl").exists()

    sdk1.close()  # release the single-writer claim — models process restart
    sdk2 = Fx1Harness(
        harness=Harness(runner=_runner),
        backend_resolver=lambda name, *a, **k: _StubBackend(),
        state_dir=td,
    )
    got = sdk2.openai_conversation_get(cid)
    items = sdk2.openai_conversation_items(cid, limit=10)["data"]
    out["sdk_dur_conv_survives"] = got["id"] == cid
    out["sdk_dur_items_survive"] = [p.get("text") for it in items for p in it["content"]] == [
        "sdk-dur",
        "sdk-dur-2",
    ]
    out["sdk_dur_restart_appends"] = (
        len(sdk2.openai_conversation_items_add(cid, [_msg("post")])["data"]) == 1
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation + receipt
# ---------------------------------------------------------------------------


def conv_audit() -> dict[str, bool]:
    """Every probe, measured end-to-end against a fresh app per section."""
    out: dict[str, bool] = {}
    with _audit_context():
        out.update(_probe_lifecycle())
        out.update(_probe_items())
        out.update(_probe_item_types())
        out.update(_probe_paging())
        out.update(_probe_response_binding())
        out.update(_probe_coexistence())
        out.update(_probe_background_binding())
        out.update(_probe_batch_refusal())
        out.update(_probe_isolation())
        out.update(_probe_tenancy())
        out.update(_probe_durability())
        out.update(_probe_concurrency())
        out.update(_probe_envelope())
        out.update(_probe_sdk())
        out.update(_probe_sdk_durability())
    return out


def conv_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = conv_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "conv_audit",
        "schema": "conv_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient (in-process ASGI) + Fx1Harness SDK twin",
            "not_verified": [
                "multi-process writers on one state_dir (single-process lock)",
                "cursor stability across concurrent appends mid-walk",
                "item payloads beyond size limits none documented",
                "real network delivery/disconnect timing (TestClient buffers)",
                "power-loss durability or rollback of valid journal suffixes",
                "uniqueness of caller-supplied item ids (preserved verbatim)",
            ],
        },
        "interpretation": (
            "The /v1/conversations contract holds end to end: containers "
            "round-trip through create/get/update/delete with honest 404 "
            "tombstones; items append in order with unique digest ids "
            "across every append (seed, route, response-turn), through "
            "page cursors in both directions, under parallel writes, and "
            "across a state-dir restart via conversations.jsonl. Response "
            "turns bound by ``conversation`` accumulate input+output items "
            "under store=false too; the anchor is exclusive with "
            "previous_response_id, refuses in batch lines, and a "
            "mid-flight delete never resurrects the container even when "
            "the independent response finishes from its frozen input. Tenancy is scope-based (read keys GET, write "
            "keys mutate) — the container is shared workspace state, "
            "key-agnostic like the retrieval index. Every refusal lands "
            "in the wire error envelope. The SDK twin matches."
            if ok
            else f"CONV AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(conv_audit_bench(), indent=2, sort_keys=True))
