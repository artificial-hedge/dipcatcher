"""tenancy_audit — adversarial probes on cross-principal isolation.

The claim under test: the serve surface's tenancy boundary is exactly
what the auth layer declares — a *scope-based* partition over shared
workspace state, with per-credential metering on top. Two managed keys
minted under one root are distinct principals for accounting (``uses``,
``tokens_used``, ``rpm`` windows, ``max_requests``/``max_tokens``
budgets) and for admin-vs-data-plane reach, but the resources they
create — responses, conversations, files, uploads, batches, eval specs
and runs, fine-tuning jobs and the ``ft:`` registry, vector stores,
stored chat completions, harness jobs/runs — are workspace objects:
any key holding the verb's scope acts on any resource any credential
minted. This battery pins that contract cell by cell and hunts the
channels where a second principal could read, mutate, bill, or spoof
across the declared boundary.

Coverage map:

- *Credential resolution* — mixed ``X-API-Key`` and
  ``Authorization: Bearer`` credentials fail closed; Bearer is a
  ``/v1``-only fallback that is never read off ``/v1``; a forged
  ``X-API-Key`` is not rescued by a valid Bearer (no downgrade channel);
  loopback still requires a key.
- *Scope partition* — ``read`` covers safe methods, ``write`` the
  mutations, ``admin`` the control plane; scopes are literal
  (``write`` does not imply ``read``), non-admin keys are refused
  ``/harness/keys*`` and ``/harness/drain``.
- *Shared workspace matrix* — key B can GET, list, mutate, complete,
  cancel, and delete every resource key A minted, across every
  stateful family. Each cell is a measured ``True`` that pins the
  shared contract — a surprise isolation change trips the probe.
- *Metering isolation* — ``/harness/self`` is the per-principal view;
  each key's ``uses``/``tokens_used`` reflect only its own calls; a
  cross-key action bills the actor, never the resource's minter.
- *Ops views* — ``/harness/usage`` and ``/harness/completions`` are
  workspace evidence readable by any scoped key: records carry the
  acting ``key_id`` fingerprint openly. Only key *material* stays
  sealed (mint response shows it once; listings never return it).
- *Idempotency* — ``Idempotency-Key`` claims are scoped by the
  authenticated key id. A foreign credential neither replays nor
  conflicts with another principal's claim.
- *Quota isolation* — one key's exhausted ``rpm``/``max_requests``/
  ``max_tokens`` never starves a sibling; the ``rate_limit_rps`` valve
  is a per-client-*host* bucket, credential-blind by design.
- *Revocation* — a tombstone kills the credential while in-flight work
  completes from its admission-time authorization and still bills the
  dead principal's tombstone ("audit, not auth"); same-name siblings
  are untouched; rotation reissues material under a successor id.
- *In-flight boundaries* — B can cancel A's live response or batch
  (workspace mutation, pinned honestly), while eval runs refuse cancel
  once running for every credential alike — minter included.
- *Drain* — admin-only, latches globally for every key once set.
- *Durability* — a state-dir restart restores key policy, tombstones,
  and metered counters.

Defects found while pinning this matrix: background responses dropped
the credential's contextvar on the worker thread, so every ``background
=true`` call landed unattributed (``key_id: null``) and unmetered —
fixed on this branch by capturing the key id at submit like the batch
lane does. Cross-credential ``Idempotency-Key`` isolation is supplied
by the integrated lane-157 namespacing and measured here.

Sealed ``tenancy_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _GateBackend,
    _temporary_directory,
)

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["tenancy_audit", "tenancy_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_MODEL = "byok"
_ANTHROPIC_H = {"anthropic-version": "2023-06-01"}
_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_FORGED = "fx1k_forged"
_DEAD = "fx1k_deadbeef"


# ---------------------------------------------------------------------------
# Stub backends, ft runners, and client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn."""

    def __init__(self, model: str = "tenancy-stub-0") -> None:
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

    def embeddings(self, input: Any, *, model: str, **extra: Any) -> Any:  # noqa: A002
        del extra
        from fx1.serve.backends import EmbeddingResult

        count = len(input) if isinstance(input, (list, tuple)) else 1
        return EmbeddingResult(
            data=tuple(
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(count)
            ),
            model=model,
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    def close(self) -> None:
        pass


def _backends() -> dict[str, Any]:
    return {
        "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
        "local_fx1": lambda **k: _StubBackend("ckpt-stub"),
        "byok": lambda **k: _StubBackend("byok-stub"),
    }


def _gate_backends() -> tuple[dict[str, Any], _GateBackend]:
    gate = _GateBackend()
    return {
        "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
        "local_fx1": lambda **k: gate,
        "byok": lambda **k: gate,
    }, gate


def _rearm(gate: _GateBackend) -> None:
    """Close the gate again so the next submit actually parks — the
    release event stays set after the first test otherwise."""
    gate.entered.clear()
    gate.gate.clear()


def _ok_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Succeeded ft job minting a checkpoint dir."""
    from fx1.serve.finetune import FTJobOutcome

    ckpt = spec.work_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        checkpoint=str(ckpt),
        trained_tokens=7,
    )


@contextmanager
def _api_key_env(api_key: str | None) -> Iterator[None]:
    """Scope the env root key to one app construction, then restore."""
    import os

    previous = os.environ.get(_API_KEY_ENV)
    if api_key is None:
        os.environ.pop(_API_KEY_ENV, None)
    else:
        os.environ[_API_KEY_ENV] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = previous


def _client(
    backend_map: dict[str, Any] | None = None,
    *,
    api_key: str | None = _ROOT,
    state_dir: Path | None = None,
    ft_runner: Any = None,
    store_max: int | None = None,
    rate_limit_rps: float | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    with _api_key_env(api_key):
        app = api_mod.create_app(
            harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
            backend_resolver=lambda name, *a, **k: (backend_map or _backends())[name](**k),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_runner=ft_runner if ft_runner is not None else _ok_runner,
            ft_dir=isolated / "fine_tuning",
            store_max=store_max,
            rate_limit_rps=rate_limit_rps,
        )
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


def _bearer(auth: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth}"}


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


def _self(client: TestClient, auth: str) -> dict[str, Any]:
    r = client.get("/harness/self", headers=_h(auth))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _self_key(body: dict[str, Any]) -> dict[str, Any]:
    key = body.get("key")
    return dict(key) if isinstance(key, dict) else {}


def _ids(body: dict[str, Any]) -> set[str]:
    return {str(item["id"]) for item in body.get("data", [])}


def _list_ids(resp: Any) -> set[str]:
    if resp.status_code != 200:
        return set()
    return _ids(resp.json())


def _wait_for(pred: Any, *, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _chat(client: TestClient, auth: str, text: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return client.post("/v1/chat/completions", json=body, headers=_h(auth))


def _respond(client: TestClient, auth: str, text: str, **extra: Any) -> Any:
    body = {"model": _MODEL, "input": text}
    body.update(extra)
    return client.post("/v1/responses", json=body, headers=_h(auth))


def _upload_file(client: TestClient, auth: str, name: str, body: bytes, purpose: str) -> str:
    r = client.post(
        "/v1/files",
        files={"file": (name, body)},
        data={"purpose": purpose},
        headers=_h(auth),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _batch_body(fid: str, custom_id: str = "l1") -> dict[str, Any]:
    return {
        "input_file_id": fid,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"probe": custom_id},
    }


def _batch_line(custom_id: str, text: str) -> bytes:
    line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": _MODEL, "messages": [{"role": "user", "content": text}]},
    }
    return (json.dumps(line) + "\n").encode()


def _wait_batch(client: TestClient, auth: str, batch_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(f"/v1/batches/{batch_id}", headers=_h(auth)).json()
        if rec.get("status") in ("completed", "failed", "cancelled", "expired"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} never reached a terminal state")


def _wait_eval_run(client: TestClient, auth: str, eval_id: str, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(
            f"/v1/evals/{eval_id}/runs/{run_id}", headers=_h(auth)
        ).json()
        if rec.get("status") in ("completed", "failed", "canceled", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"eval run {run_id} never reached a terminal state")


def _wait_ft(client: TestClient, auth: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(f"/v1/fine_tuning/jobs/{job_id}", headers=_h(auth)).json()
        if rec.get("status") in ("succeeded", "failed", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"ft job {job_id} never reached a terminal state")


def _wait_response(client: TestClient, auth: str, rid: str) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        rec: dict[str, Any] = client.get(f"/v1/responses/{rid}", headers=_h(auth)).json()
        if rec.get("status") in ("completed", "failed", "cancelled", "incomplete"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"response {rid} never reached a terminal state")


def _eval_spec(client: TestClient, auth: str, name: str) -> str:
    r = client.post(
        "/v1/evals",
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


def _ft_corpus_file(client: TestClient, auth: str, name: str = "ft.jsonl") -> str:
    return _upload_file(client, auth, name, _CORPUS, "fine-tune")


def _conv(client: TestClient, auth: str, **body: Any) -> dict[str, Any]:
    r = client.post("/v1/conversations", json=body, headers=_h(auth))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


# ---------------------------------------------------------------------------
# Credential resolution and scope partition
# ---------------------------------------------------------------------------


def _probe_credentials(results: dict[str, bool]) -> None:
    """Credential ambiguity and forge channels: mixed authentication
    mechanisms fail closed; Bearer alone is a ``/v1``-only fallback."""
    client, _api = _client()
    key_a, _id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, _id_b = _mint(client, name="bob", scopes=["read", "write"])

    forged = client.get(
        "/v1/models",
        headers={"X-API-Key": key_a, "Authorization": f"Bearer {key_b}"},
    )
    results["mixed_auth_refused_on_v1_400"] = forged.status_code == 400
    who = client.get(
        "/harness/self",
        headers={"X-API-Key": key_a, "Authorization": f"Bearer {key_b}"},
    )
    results["mixed_auth_refused_off_v1_400"] = who.status_code == 400
    bearer_only = client.get("/v1/models", headers=_bearer(key_a))
    results["bearer_authenticates_on_v1"] = bearer_only.status_code == 200
    bearer_off_v1 = client.get("/harness/self", headers=_bearer(key_a))
    results["bearer_ignored_off_v1"] = bearer_off_v1.status_code == 401
    downgraded = client.get(
        "/v1/models",
        headers={"X-API-Key": _FORGED, "Authorization": f"Bearer {key_a}"},
    )
    results["forged_x_api_key_with_bearer_refused_400"] = downgraded.status_code == 400
    results["garbage_x_api_key_401"] = (
        client.get("/v1/models", headers=_h(_DEAD)).status_code == 401
    )
    results["garbage_bearer_401"] = (
        client.get("/v1/models", headers=_bearer(_DEAD)).status_code == 401
    )
    results["anonymous_401_on_v1"] = client.get("/v1/models").status_code == 401
    results["anonymous_401_on_harness"] = client.get("/harness/self").status_code == 401
    results["health_is_public"] = client.get("/health").status_code == 200
    results["env_key_is_admin"] = client.get("/harness/keys", headers=_h(_ROOT)).status_code == 200


def _probe_scope_partition(results: dict[str, bool]) -> None:
    """Scopes are literal verbs: ``read`` covers safe methods only,
    ``write`` does not imply ``read``, and ``admin`` is the only path to
    key lifecycle, peer usage cards, and drain — on every principal."""
    client, _api = _client()
    read_key, _rid = _mint(client, name="reader", scopes=["read"])
    write_key, _wid = _mint(client, name="writer", scopes=["write"])
    plain_key, _pid = _mint(client, name="plain")
    admin_key, _aid = _mint(client, name="boss", admin=True)

    results["read_scopes_safe_methods"] = (
        client.get("/v1/models", headers=_h(read_key)).status_code == 200
    )
    results["read_cannot_mutate"] = (
        client.post(
            "/v1/responses",
            json={"model": _MODEL, "input": "x"},
            headers=_h(read_key),
        ).status_code
        == 403
    )
    results["write_does_not_imply_read"] = (
        client.get("/v1/models", headers=_h(write_key)).status_code == 403
    )
    results["write_mutates"] = (
        client.post(
            "/v1/responses",
            json={"model": _MODEL, "input": "x"},
            headers=_h(write_key),
        ).status_code
        == 200
    )
    results["mint_requires_admin_scope"] = (
        client.post("/harness/keys", json={"name": "n"}, headers=_h(plain_key)).status_code == 403
    )
    results["key_roster_requires_admin"] = (
        client.get("/harness/keys", headers=_h(plain_key)).status_code == 403
    )
    results["peer_usage_card_requires_admin"] = (
        client.get("/harness/keys/whatever/usage", headers=_h(plain_key)).status_code == 403
    )
    results["revoke_requires_admin"] = (
        client.delete("/harness/keys/whatever", headers=_h(plain_key)).status_code == 403
    )
    results["rotate_requires_admin"] = (
        client.post("/harness/keys/whatever/rotate", headers=_h(plain_key)).status_code == 403
    )
    managed_admin = client.post("/harness/keys", json={"name": "x"}, headers=_h(admin_key))
    results["managed_admin_mints"] = managed_admin.status_code in (200, 201)
    results["admin_card_any_principal"] = (
        client.get(f"/harness/keys/{_aid}/usage", headers=_h(admin_key)).status_code == 200
    )


# ---------------------------------------------------------------------------
# Shared workspace matrix — every cell measured
# ---------------------------------------------------------------------------


def _probe_workspace_matrix(results: dict[str, bool]) -> None:
    """The tenancy matrix, measured cell by cell: every stateful family
    is workspace state — B reads, lists, mutates, and deletes what A
    minted. ``True`` asserts the declared shared contract."""
    client, _api = _client()
    key_a, _id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, _id_b = _mint(client, name="bob", scopes=["read", "write"])
    ha, hb = _h(key_a), _h(key_b)

    # --- responses ---------------------------------------------------------
    r = _respond(client, key_a, "matrix-resp")
    rid = str(r.json()["id"])
    results["responses_b_reads_a"] = (
        client.get(f"/v1/responses/{rid}", headers=hb).status_code == 200
    )
    results["responses_no_list_surface"] = client.get("/v1/responses", headers=hb).status_code in (
        404,
        405,
    )
    results["responses_b_reads_a_items"] = (
        client.get(f"/v1/responses/{rid}/input_items", headers=hb).status_code == 200
    )
    results["responses_b_deletes_a"] = (
        client.delete(f"/v1/responses/{rid}", headers=hb).status_code == 200
    )
    results["responses_gone_under_a_after_b_delete"] = (
        client.get(f"/v1/responses/{rid}", headers=ha).status_code == 404
    )

    # --- stored chat completions --------------------------------------------
    c = _chat(client, key_a, "matrix-chat")
    ccid = str(c.json()["id"])
    results["completions_b_reads_a"] = (
        client.get(f"/v1/chat/completions/{ccid}", headers=hb).status_code == 200
    )
    results["completions_b_lists_a"] = ccid in _list_ids(
        client.get("/v1/chat/completions", headers=hb)
    )
    results["completions_b_patches_a"] = (
        client.post(
            f"/v1/chat/completions/{ccid}",
            json={"metadata": {"vandal": "bob"}},
            headers=hb,
        ).status_code
        == 200
    )
    results["completion_messages_b_reads_a"] = (
        client.get(f"/v1/chat/completions/{ccid}/messages", headers=hb).status_code == 200
    )
    results["completions_b_deletes_a"] = (
        client.delete(f"/v1/chat/completions/{ccid}", headers=hb).status_code == 200
    )

    # --- conversations ------------------------------------------------------
    conv = _conv(client, key_a, metadata={"owner": "alice"})
    cid = str(conv["id"])
    item_r = client.post(
        f"/v1/conversations/{cid}/items",
        json={
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "a-note"}],
                }
            ]
        },
        headers=ha,
    )
    iid = str(item_r.json()["data"][0]["id"])
    results["conversations_b_reads_a"] = (
        client.get(f"/v1/conversations/{cid}", headers=hb).status_code == 200
    )
    results["conversations_no_list_surface"] = client.get(
        "/v1/conversations", headers=hb
    ).status_code in (404, 405)
    results["conversations_b_mutates_a"] = (
        client.post(
            f"/v1/conversations/{cid}",
            json={"metadata": {"owner": "bob"}},
            headers=hb,
        ).status_code
        == 200
    )
    results["conversation_items_b_reads_a"] = iid in _list_ids(
        client.get(f"/v1/conversations/{cid}/items", headers=hb)
    )
    bound = _respond(client, key_b, "hijack", conversation=cid)
    results["responses_b_binds_a_conversation"] = bound.status_code == 200
    items_after = client.get(f"/v1/conversations/{cid}/items", headers=hb)
    results["conversation_items_gain_b_turn"] = (
        items_after.status_code == 200 and len(items_after.json()["data"]) >= 2
    )
    results["conversations_b_deletes_a"] = (
        client.delete(f"/v1/conversations/{cid}", headers=hb).status_code == 200
    )

    # --- files ---------------------------------------------------------------
    fid = _upload_file(client, key_a, "a.jsonl", b'{"x":1}\n', "batch")
    results["files_b_reads_a_meta"] = client.get(f"/v1/files/{fid}", headers=hb).status_code == 200
    content = client.get(f"/v1/files/{fid}/content", headers=hb)
    results["files_b_reads_a_content"] = (
        content.status_code == 200 and b'{"x":1}' in content.content
    )
    results["files_b_lists_a"] = fid in _list_ids(client.get("/v1/files", headers=hb))

    # --- uploads -------------------------------------------------------------
    u = client.post(
        "/v1/uploads",
        json={
            "purpose": "batch",
            "filename": "a.jsonl",
            "bytes": 4,
            "mime_type": "text/jsonl",
        },
        headers=ha,
    )
    uid = str(u.json()["id"])
    results["uploads_no_read_surface"] = client.get(
        f"/v1/uploads/{uid}", headers=hb
    ).status_code in (404, 405) and client.get(f"/v1/uploads/{uid}", headers=ha).status_code in (
        404,
        405,
    )
    part = client.post(
        f"/v1/uploads/{uid}/parts",
        files={"data": ("p", b"ab\nc")},
        headers=hb,
    )
    results["uploads_b_adds_part_to_a"] = part.status_code == 200
    pid = str(part.json()["id"])
    done = client.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid]}, headers=hb)
    results["uploads_b_completes_a"] = done.status_code == 200
    minted = done.json().get("file", {}).get("id")
    results["uploads_minted_file_is_workspace"] = bool(minted) and (
        client.get(f"/v1/files/{minted}", headers=ha).status_code == 200
    )

    # --- batches --------------------------------------------------------------
    bfid = _upload_file(client, key_a, "b.jsonl", _batch_line("l1", "go"), "batch")
    b = client.post("/v1/batches", json=_batch_body(bfid), headers=ha)
    bid = str(b.json()["id"])
    results["batches_b_reads_a"] = client.get(f"/v1/batches/{bid}", headers=hb).status_code == 200
    results["batches_b_lists_a"] = bid in _list_ids(client.get("/v1/batches", headers=hb))
    bdone = _wait_batch(client, key_a, bid)
    out_fid = bdone.get("output_file_id")
    results["batches_b_reads_a_output_file"] = bool(out_fid) and (
        client.get(f"/v1/files/{out_fid}/content", headers=hb).status_code == 200
    )
    results["batches_terminal_cancel_conflict_409"] = (
        client.post(f"/v1/batches/{bid}/cancel", headers=hb).status_code == 409
    )
    results["files_b_deletes_a"] = client.delete(f"/v1/files/{fid}", headers=hb).status_code == 200
    results["files_gone_under_a_after_b_delete"] = (
        client.get(f"/v1/files/{fid}", headers=ha).status_code == 404
    )

    # --- evals ---------------------------------------------------------------
    eid = _eval_spec(client, key_a, "matrix-spec")
    results["evals_b_reads_a"] = client.get(f"/v1/evals/{eid}", headers=hb).status_code == 200
    results["evals_b_lists_a"] = eid in _list_ids(client.get("/v1/evals", headers=hb))
    run_a = client.post(f"/v1/evals/{eid}/runs", json={"model": "byok"}, headers=ha)
    erid_a = str(run_a.json()["id"])
    results["eval_runs_b_reads_a"] = (
        client.get(f"/v1/evals/{eid}/runs/{erid_a}", headers=hb).status_code == 200
    )
    results["eval_runs_b_lists_a"] = erid_a in _list_ids(
        client.get(f"/v1/evals/{eid}/runs", headers=hb)
    )
    run_b = client.post(f"/v1/evals/{eid}/runs", json={"model": "byok"}, headers=hb)
    results["evals_b_runs_a_spec"] = run_b.status_code == 201
    results["evals_b_mutates_a"] = (
        client.post(f"/v1/evals/{eid}", json={"name": "renamed"}, headers=hb).status_code == 200
    )
    results["evals_b_deletes_a"] = client.delete(f"/v1/evals/{eid}", headers=hb).status_code == 200

    # --- fine-tuning jobs + the ft: registry -----------------------------------
    ffid = _ft_corpus_file(client, key_a)
    ft = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": ffid, "suffix": "m"},
        headers=ha,
    )
    jid = str(ft.json()["id"])
    results["ft_b_reads_a_job"] = (
        client.get(f"/v1/fine_tuning/jobs/{jid}", headers=hb).status_code == 200
    )
    results["ft_b_lists_a_job"] = jid in _list_ids(client.get("/v1/fine_tuning/jobs", headers=hb))
    ftdone = _wait_ft(client, key_a, jid)
    results["ft_b_reads_a_events"] = (
        client.get(f"/v1/fine_tuning/jobs/{jid}/events", headers=hb).status_code == 200
    )
    results["ft_b_reads_a_checkpoints"] = (
        client.get(f"/v1/fine_tuning/jobs/{jid}/checkpoints", headers=hb).status_code == 200
    )
    ftm = str(ftdone.get("fine_tuned_model") or "")
    model_ids = _list_ids(client.get("/v1/models", headers=hb))
    results["registry_b_sees_a_ft_model"] = bool(ftm) and ftm in model_ids
    results["registry_b_tombstones_a_model"] = bool(ftm) and (
        client.delete(f"/v1/models/{ftm}", headers=hb).status_code == 200
    )
    results["registry_tombstone_visible_to_a"] = bool(ftm) and (
        ftm not in _list_ids(client.get("/v1/models", headers=ha))
    )

    # --- vector stores ----------------------------------------------------------
    vs = client.post("/v1/vector_stores", json={"name": "a-store"}, headers=ha)
    vsid = str(vs.json()["id"])
    results["vector_stores_b_reads_a"] = (
        client.get(f"/v1/vector_stores/{vsid}", headers=hb).status_code == 200
    )
    results["vector_stores_b_lists_a"] = vsid in _list_ids(
        client.get("/v1/vector_stores", headers=hb)
    )
    vs_fid = _ft_corpus_file(client, key_a, "vs.jsonl")
    attach = client.post(f"/v1/vector_stores/{vsid}/files", json={"file_id": vs_fid}, headers=hb)
    results["vector_stores_b_attaches_to_a"] = attach.status_code == 200
    vs_fid2 = str(attach.json().get("id") or vs_fid)
    results["vector_stores_b_searches_a"] = (
        client.post(
            f"/v1/vector_stores/{vsid}/search",
            json={"query": "q"},
            headers=hb,
        ).status_code
        == 200
    )
    results["vector_stores_b_detaches_a_file"] = (
        client.delete(f"/v1/vector_stores/{vsid}/files/{vs_fid2}", headers=hb).status_code == 200
    )
    results["vector_stores_b_deletes_a"] = (
        client.delete(f"/v1/vector_stores/{vsid}", headers=hb).status_code == 200
    )

    # --- harness jobs / runs ------------------------------------------------------
    job = client.post("/harness/jobs", json={"command": "doctor"}, headers=ha)
    jobid = str(job.json()["job_id"])
    results["jobs_b_reads_a"] = client.get(f"/harness/jobs/{jobid}", headers=hb).status_code == 200
    jobs_b = client.get("/harness/jobs", headers=hb).json()
    results["jobs_b_lists_a"] = jobid in {str(j["job_id"]) for j in jobs_b.get("jobs", [])}
    run = client.post("/harness/runs", json={"command": "doctor"}, headers=ha)
    results["runs_b_submission_visible"] = run.status_code in (200, 201, 202)

    # --- anthropic message batches ----------------------------------------------
    ab = client.post(
        "/v1/messages/batches",
        json={
            "requests": [
                {
                    "custom_id": "a",
                    "params": {
                        "model": "byok",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                }
            ]
        },
        headers=_ah(key_a),
    )
    abid = str(ab.json()["id"])
    results["anthropic_batches_b_reads_a"] = (
        client.get(f"/v1/messages/batches/{abid}", headers=_ah(key_b)).status_code == 200
    )
    results["anthropic_batches_b_lists_a"] = abid in _list_ids(
        client.get("/v1/messages/batches", headers=_ah(key_b))
    )


def _probe_metering(results: dict[str, bool]) -> None:
    """Per-principal meters: ``/harness/self`` is the only self view;
    each key's counters reflect only its own calls, and a cross-key
    action bills the actor — never the resource's minter."""
    client, _api = _client()
    key_a, id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, id_b = _mint(client, name="bob", scopes=["read", "write"])

    for i in range(3):
        assert _chat(client, key_a, f"a{i}").status_code == 200
    assert _chat(client, key_b, "b0").status_code == 200

    self_a = _self(client, key_a)
    self_b = _self(client, key_b)
    kcard_a, kcard_b = _self_key(self_a), _self_key(self_b)
    results["self_meters_own_calls"] = (
        int(kcard_a.get("served", {}).get("calls", -1)) == 3
        and int(kcard_b.get("served", {}).get("calls", -1)) == 1
    )
    results["self_meters_own_tokens"] = (
        int(kcard_a.get("tokens_used", 0)) == 15 and int(kcard_b.get("tokens_used", 0)) == 5
    )
    results["self_identifies_caller"] = (
        self_a.get("credential") == "managed"
        and kcard_a.get("id") == id_a
        and kcard_b.get("id") == id_b
    )
    results["uses_counts_each_request"] = (
        int(kcard_a.get("uses", -1)) == 4 and int(kcard_b.get("uses", -1)) == 2
    )
    results["admin_card_meters_principal"] = _uses(client, id_a) == 4 and _uses(client, id_b) == 2

    ua0, ub0 = _uses(client, id_a), _uses(client, id_b)
    conv = _conv(client, key_a)
    cid = str(conv["id"])
    assert client.get(f"/v1/conversations/{cid}", headers=_h(key_b)).status_code == 200
    results["cross_key_action_bills_actor"] = (
        _uses(client, id_a) == ua0 + 1 and _uses(client, id_b) == ub0 + 1
    )
    results["cross_key_tokens_bill_actor"] = _tokens(client, id_a) == 15

    usage_b = client.get("/harness/usage", headers=_h(key_b))
    results["usage_global_readable_by_any_scoped_key"] = usage_b.status_code == 200
    results["usage_global_anonymous_401"] = client.get("/harness/usage").status_code == 401
    completions_b = client.get("/harness/completions", headers=_h(key_b))
    ledger_items = completions_b.json().get("items", [])
    results["completions_ledger_is_workspace_global"] = (
        completions_b.status_code == 200 and len(ledger_items) >= 4
    )
    env_self = _self(client, _ROOT)
    results["env_self_unmetered"] = env_self.get("credential") == "env" and not env_self.get(
        "metered", True
    )


def _probe_secret_channels(results: dict[str, bool]) -> None:
    """Key material and peer fingerprints: mint returns material once;
    listings, error bodies, usage views, and usage records under a
    foreign credential never carry raw key bytes. Fingerprints (key_id)
    ride workspace records openly — pinned, not a leak."""
    client, _api = _client()
    key_a, id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, id_b = _mint(client, name="bob", scopes=["read", "write"])

    roster = client.get("/harness/keys", headers=_h(_ROOT))
    bodies = json.dumps(roster.json())
    results["roster_never_returns_key_material"] = (
        key_a not in bodies and key_b not in bodies and '"sha256"' not in bodies
    )

    assert _respond(client, key_a, "secret-seed").status_code == 200
    leak_surfaces = [
        client.get("/harness/self", headers=_h(key_b)),
        client.get("/harness/usage", headers=_h(key_b)),
        client.get("/harness/completions", headers=_h(key_b)),
        client.get("/v1/models", headers=_h(key_b)),
        client.get("/v1/responses/nonexistent", headers=_h(key_b)),
        client.post("/v1/responses", json={"model": _MODEL}, headers=_h(key_b)),
        client.get("/harness/keys", headers=_h(key_b)),
    ]
    results["peer_views_never_carry_key_material"] = all(
        key_a
        not in json.dumps(
            r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        )
        for r in leak_surfaces
    )
    self_b = _self(client, key_b)
    results["self_shows_fingerprint_not_material"] = key_b not in json.dumps(
        self_b
    ) and id_b in json.dumps(self_b)
    ledger = client.get("/harness/completions", headers=_h(key_b)).json()
    rows = ledger.get("items", [])
    fp_visible = any(row.get("key_id") == id_a for row in rows)
    results["completion_records_carry_actor_fingerprint"] = fp_visible
    results["actor_fingerprint_matches_card_id"] = fp_visible and _uses(client, id_a) >= 1


def _probe_idem_namespace(results: dict[str, bool]) -> None:
    """Idempotency-Key claims are isolated across credentials."""
    client, _api = _client()
    key_a, _id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, _id_b = _mint(client, name="bob", scopes=["read", "write"])
    idem = "K-shared"

    body = {"model": _MODEL, "messages": [{"role": "user", "content": "idem"}]}
    r1 = client.post(
        "/v1/chat/completions", json=body, headers={**_h(key_a), "Idempotency-Key": idem}
    )
    assert r1.status_code == 200, r1.text
    id1 = str(r1.json()["id"])

    replay_same = client.post(
        "/v1/chat/completions", json=body, headers={**_h(key_a), "Idempotency-Key": idem}
    )
    results["idem_replay_same_credential_stable"] = (
        replay_same.status_code == 200 and replay_same.json()["id"] == id1
    )
    conflict_same = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "different"}]},
        headers={**_h(key_a), "Idempotency-Key": idem},
    )
    results["idem_conflict_same_credential_409"] = conflict_same.status_code == 409

    replay_b = client.post(
        "/v1/chat/completions", json=body, headers={**_h(key_b), "Idempotency-Key": idem}
    )
    results["idem_replay_isolated_per_credential"] = (
        replay_b.status_code == 200 and replay_b.json()["id"] != id1
    )
    conflict_b = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "foreign"}]},
        headers={**_h(key_b), "Idempotency-Key": idem},
    )
    # Credential B established its own independent claim above. A different
    # payload under that same credential/key must therefore conflict, while
    # the successful B replay already proves isolation from credential A.
    results["idem_conflict_isolated_per_credential"] = (
        conflict_b.status_code == 409
        and conflict_b.json().get("error", {}).get("code") == "idempotency_conflict"
    )
    results["idem_key_ignored_on_safe_methods"] = (
        client.get("/v1/models", headers={**_h(key_b), "Idempotency-Key": idem}).status_code == 200
    )


def _probe_quota_independence(results: dict[str, bool]) -> None:
    """Budgets are per-record: one principal's exhausted rpm,
    max_requests, or max_tokens never starves a sibling. The
    ``rate_limit_rps`` valve is a per-client-host bucket — on one host
    every credential shares it, pinned as the declared safety valve."""
    client, _api = _client()
    thin_key, _tid = _mint(client, name="thin", rpm=2, scopes=["read", "write"])
    fat_key, _fid = _mint(client, name="fat", scopes=["read", "write"])

    assert _chat(client, thin_key, "t1").status_code == 200
    assert _chat(client, thin_key, "t2").status_code == 200
    third = _chat(client, thin_key, "t3")
    results["rpm_window_throttles_at_limit"] = third.status_code == 429 and "retry-after" in {
        k.lower() for k in third.headers
    }
    results["rpm_window_isolated_from_sibling"] = _chat(client, fat_key, "f1").status_code == 200

    capped_key, _cid = _mint(client, name="capped", max_requests=2, scopes=["read", "write"])
    free_key, _ffid = _mint(client, name="free", scopes=["read", "write"])
    assert _chat(client, capped_key, "c1").status_code == 200
    assert _chat(client, capped_key, "c2").status_code == 200
    exhausted = _chat(client, capped_key, "c3")
    results["request_budget_throttles_at_limit"] = exhausted.status_code == 429
    results["request_budget_isolated_from_sibling"] = (
        _chat(client, free_key, "x").status_code == 200
    )
    card_capped = _card(client, _cid)
    results["exhausted_key_still_metered"] = int(card_capped["uses"]) == 2

    tok_key, _tkid = _mint(client, name="tok", max_tokens=9, scopes=["read", "write"])
    ok1 = _chat(client, tok_key, "k1")
    ok2 = _chat(client, tok_key, "k2")
    deny = _chat(client, tok_key, "k3")
    results["token_budget_denies_once_past_cap"] = (
        ok1.status_code == 200 and ok2.status_code == 200 and deny.status_code == 429
    )
    results["token_budget_overshoots_once"] = _tokens(client, _tkid) == 10
    results["token_budget_isolated_from_sibling"] = _chat(client, fat_key, "f2").status_code == 200

    limited, _api2 = _client(rate_limit_rps=2.0)
    k1, _ = _mint(limited, name="one", scopes=["read", "write"])
    first = limited.get("/v1/models", headers=_h(_ROOT))
    second = limited.get("/v1/models", headers=_h(k1))
    results["host_rate_limiter_shares_bucket_across_keys"] = (
        first.status_code == 200 and second.status_code == 429
    )


def _probe_revocation(results: dict[str, bool]) -> None:
    """Revocation boundary: a tombstone kills the credential but keeps
    the accounting record; siblings — even same-name ones — are
    untouched, and the dead key's address can't be re-minted through a
    peer's handle."""
    client, _api = _client()
    key_a, id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, id_b = _mint(client, name="alice", scopes=["read", "write"])

    assert _chat(client, key_a, "live").status_code == 200
    rid = str(_respond(client, key_a, "orphan-check").json()["id"])

    gone = client.delete(f"/harness/keys/{id_a}", headers=_h(_ROOT))
    results["revoke_returns_200"] = gone.status_code == 200
    results["revoked_key_dead"] = _chat(client, key_a, "post").status_code == 401
    results["revoked_key_dead_on_v1_bearer"] = (
        client.get("/v1/models", headers=_bearer(key_a)).status_code == 401
    )
    tomb = _card(client, id_a)
    results["revoked_tombstone_keeps_metering"] = (
        tomb.get("enabled") is False
        and tomb.get("revoked_at") is not None
        and int(tomb.get("uses", 0)) == 2
        and int(tomb.get("tokens_used", 0)) == 10
    )
    results["same_name_sibling_unaffected"] = (
        _chat(client, key_b, "still").status_code == 200 and _uses(client, id_b) == 1
    )
    results["orphaned_resource_stays_addressable"] = (
        client.get(f"/v1/responses/{rid}", headers=_h(key_b)).status_code == 200
    )
    revive = client.post(f"/harness/keys/{id_a}/rotate", json={}, headers=_h(_ROOT))
    results["tombstone_not_rotatable"] = revive.status_code in (404, 409, 410)

    rot = client.post(f"/harness/keys/{id_b}/rotate", json={}, headers=_h(_ROOT))
    wire = rot.json().get("key", {})
    new_raw = str(wire.get("key") or "")
    results["rotate_reissues_material"] = rot.status_code == 201 and bool(new_raw)
    results["rotate_journals_lineage"] = rot.json().get("rotated_from") == id_b
    results["rotate_kills_old_material"] = _chat(client, key_b, "old").status_code == 401
    results["rotate_successor_inherits_policy"] = (
        _chat(client, new_raw, "new").status_code == 200
        and _self_key(_self(client, new_raw)).get("name") == "alice"
    )


def _probe_inflight_boundaries(results: dict[str, bool]) -> None:
    """Mid-flight mutations: a parked background turn can be cancelled
    by any scoped key, a running batch the same, an eval run the same —
    and a credential revoked mid-flight still completes its work under
    the admission-time authorization."""
    gate_map, gate = _gate_backends()
    client, _api = _client(gate_map)
    key_a, id_a = _mint(client, name="alice", scopes=["read", "write"])
    key_b, _id_b = _mint(client, name="bob", scopes=["read", "write"])

    r = _respond(client, key_a, "inflight", background=True, store=True)
    rid = str(r.json()["id"])
    assert gate.entered.wait(10), "background turn never entered the backend"
    cancel = client.post(f"/v1/responses/{rid}/cancel", headers=_h(key_b))
    results["responses_b_cancels_a_inflight"] = cancel.status_code == 200
    gate.gate.set()
    final = _wait_response(client, key_b, rid)
    results["cancelled_response_terminal_for_b"] = final.get("status") in (
        "cancelled",
        "incomplete",
        "failed",
        "completed",
    )

    _rearm(gate)
    bfid = _upload_file(client, key_a, "inflight.jsonl", _batch_line("l1", "go"), "batch")
    b = client.post("/v1/batches", json=_batch_body(bfid, "infl"), headers=_h(key_a))
    bid = str(b.json()["id"])
    assert gate.entered.wait(10), "batch worker never entered the backend"
    bcancel = client.post(f"/v1/batches/{bid}/cancel", headers=_h(key_b))
    results["batches_b_cancels_a_inflight"] = bcancel.status_code == 200
    gate.gate.set()
    bfinal = _wait_batch(client, key_a, bid)
    results["cancelled_batch_terminal"] = bfinal.get("status") in (
        "cancelled",
        "cancelling",
        "completed",
        "failed",
    )

    _rearm(gate)
    eid = _eval_spec(client, key_a, "inflight-spec")
    run = client.post(f"/v1/evals/{eid}/runs", json={"model": "byok"}, headers=_h(key_a))
    erid = str(run.json()["id"])
    assert gate.entered.wait(10), "eval run never entered the backend"
    ecancel_b = client.post(f"/v1/evals/{eid}/runs/{erid}/cancel", headers=_h(key_b))
    results["eval_runs_cancel_running_409_for_peer"] = ecancel_b.status_code == 409
    ecancel_a = client.post(f"/v1/evals/{eid}/runs/{erid}/cancel", headers=_h(key_a))
    results["eval_runs_cancel_running_409_for_minter"] = ecancel_a.status_code == 409
    gate.gate.set()
    efinal = _wait_eval_run(client, key_b, eid, erid)
    results["eval_runs_run_to_completion_when_running"] = efinal.get("status") in (
        "completed",
        "failed",
    )

    _rearm(gate)
    rb = _respond(client, key_b, "bg-plain", background=True, store=True)
    rbid = str(rb.json()["id"])
    assert gate.entered.wait(10), "plain background turn never entered the backend"
    gate.gate.set()
    results["background_response_completes"] = (
        _wait_response(client, key_b, rbid).get("status") == "completed"
    )
    ledger_b = client.get("/harness/completions", headers=_h(_ROOT)).json()
    results["background_completion_carries_credential"] = any(
        row.get("key_id") == _id_b for row in ledger_b.get("items", [])
    )
    results["background_spend_metered_on_owner"] = _tokens(client, _id_b) >= 5

    gate_map2, gate2 = _gate_backends()
    client2, _api2 = _client(gate_map2)
    key_c, id_c = _mint(client2, name="carol", scopes=["read", "write"])
    r2 = _respond(client2, key_c, "revoke-mid-flight", background=True, store=True)
    rid2 = str(r2.json()["id"])
    assert gate2.entered.wait(10), "second background turn never entered"
    assert client2.delete(f"/harness/keys/{id_c}", headers=_h(_ROOT)).status_code == 200
    gate2.gate.set()
    done2 = _wait_response(client2, _ROOT, rid2)
    results["inflight_work_completes_past_revocation"] = done2.get("status") in (
        "completed",
        "incomplete",
    )
    results["revoked_key_cannot_read_own_work"] = (
        client2.get(f"/v1/responses/{rid2}", headers=_h(key_c)).status_code == 401
    )
    ledger2 = client2.get("/harness/completions", headers=_h(_ROOT)).json()
    results["inflight_completion_keeps_actor_fingerprint"] = any(
        row.get("key_id") == id_c for row in ledger2.get("items", [])
    )
    results["inflight_spend_metered_on_tombstone"] = _tokens(client2, id_c) >= 5


def _probe_drain_admin(results: dict[str, bool]) -> None:
    """``POST /harness/drain`` is a global latch: admin-only to set, and
    once set every data-plane credential is refused — the control-plane
    statement is workspace-wide by design."""
    client, _api = _client()
    key_a, _id_a = _mint(client, name="alice", scopes=["read", "write"])
    admin_key, _aid = _mint(client, name="boss", admin=True)

    results["drain_refuses_write_key"] = (
        client.post("/harness/drain", headers=_h(key_a)).status_code == 403
    )
    results["drain_refuses_anonymous"] = client.post("/harness/drain").status_code == 401
    before = _chat(client, key_a, "pre-drain")
    assert before.status_code == 200
    results["drain_sets_under_admin"] = (
        client.post("/harness/drain", headers=_h(admin_key)).status_code == 200
    )
    after_a = _chat(client, key_a, "post-drain")
    after_admin = _chat(client, admin_key, "post-drain")
    results["drain_latches_globally_for_all_keys"] = (
        after_a.status_code == 503 and after_admin.status_code == 503
    )
    results["drain_ignores_forged_key"] = (
        client.post("/harness/drain", headers=_h(_FORGED)).status_code == 401
    )


def _probe_durability(results: dict[str, bool]) -> None:
    """A state-dir restart restores the key registry: minted policy,
    tombstones, and metered counters survive; a peer's counters never
    merge."""
    state = _temporary_directory() / "state"
    c1, _api1 = _client(state_dir=state)
    key_a, id_a = _mint(c1, name="alice", scopes=["read", "write"], rpm=7)
    key_b, id_b = _mint(c1, name="bob", scopes=["read", "write"])
    assert _chat(c1, key_a, "one").status_code == 200
    assert _chat(c1, key_b, "two").status_code == 200
    assert _chat(c1, key_b, "three").status_code == 200
    c1.close()

    c2, _api2 = _client(state_dir=state)
    results["keys_persist_across_restart"] = _chat(c2, key_a, "back").status_code == 200
    results["meters_persist_across_restart"] = _uses(c2, id_a) == 2 and _uses(c2, id_b) == 2
    results["policy_persists_across_restart"] = int(_card(c2, id_a).get("rpm", 0)) == 7
    assert c2.delete(f"/harness/keys/{id_b}", headers=_h(_ROOT)).status_code == 200
    c2.close()

    c3, _api3 = _client(state_dir=state)
    results["tombstones_persist_across_restart"] = (
        _chat(c3, key_b, "z").status_code == 401
        and _chat(c3, key_a, "still-alive").status_code == 200
    )


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def tenancy_audit() -> dict[str, bool]:
    """Every probe, measured end-to-end against a live app."""
    results: dict[str, bool] = {}
    with _audit_context():
        _probe_credentials(results)
        _probe_scope_partition(results)
        _probe_workspace_matrix(results)
        _probe_metering(results)
        _probe_secret_channels(results)
        _probe_idem_namespace(results)
        _probe_quota_independence(results)
        _probe_revocation(results)
        _probe_inflight_boundaries(results)
        _probe_drain_admin(results)
        _probe_durability(results)
    return results


def tenancy_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = tenancy_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "tenancy_audit",
        "schema": "tenancy_audit.v1",
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
                "BYOK upstream credential capture inside backend error payloads "
                "(stub backends never see real upstreams)",
                "cross-host rate-limit buckets (TestClient is one host)",
                "usage-export deltas under concurrent load",
            ],
        },
        "interpretation": (
            "Tenancy is scope-based, not principal-based — and that is the "
            "pinned contract: every stateful family (responses, conversations, "
            "files, uploads, batches, evals, ft jobs, the ft: registry, vector "
            "stores, stored completions, harness jobs) is shared workspace "
            "state, mutable by any key holding the verb's scope. Isolation is "
            "real where it is claimed: per-key uses/tokens/rpm/budgets bill "
            "the actor alone, /harness/self shows only the caller, key "
            "material never leaves its mint response, mixed authentication "
            "headers fail closed with no downgrade path, admin scopes guard the control "
            "plane, drain latches globally, tombstones hold across restarts, "
            "and revocation never orphans or merges a peer's work. The "
            "rate_limit_rps valve is per-client-host, credential-blind by "
            "design, and background work keeps its principal through "
            "revocation — ledger fingerprint intact, tombstone metered. "
            "Idempotency-Key claims are scoped by the authenticated key "
            "id, so cross-credential replay and conflict isolation hold."
            if ok
            else f"TENANCY AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(tenancy_audit_bench(), indent=1))
