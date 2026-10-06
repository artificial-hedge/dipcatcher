"""drain_audit — ``POST /harness/drain`` lifecycle deep audit.

Probe battery over the drain-shutdown contract: the latch itself
(admin scope, honest body, one-way idempotent re-latch, ``wait_s``
window semantics, ``/ready`` deregistration signal, metrics gauge),
the refusal set (every mutating surface — model calls, job/eval/batch
submits and stored-resource writes — answers
``503 draining`` in its own error grammar), the read-through set
(every stored resource's list/get plus cancels, deletes, pauses, and
the documented advisory preflights stay open), in-flight honesty (a
job/run/response/stream admitted before the latch completes to a real
terminal state), the submit-then-drain race boundary, recovery (no
resume route; the latch is process-local — a ``--state-dir`` restart
clears it and re-arms), drain-time webhooks (a pending callback still
fires once, signed), key store behavior under drain (existing keys
still authenticate reads and emergency revocation remains open), metering (a
drain-refused call still bills ``uses`` — admission happens at
authenticate, before the gate — while auth/scope refusals bill
nothing), idempotency (a refused submit leaves no phantom claim; a
stored replay still serves under drain), envelope uniformity
(``draining`` code on the /harness + OpenAI grammars, ``api_error``
type on the Anthropic grammar), and cascades (a refused batch submit
consumes no lines; a refused submit leaves no store records).

Found while building this lane (fixed in the same commit):

* The drain latch covered only slot-admitting routes — the whole
  stored-resource mutation surface leaked: ``/v1/files``, the
  ``/v1/uploads`` create/parts/complete lifecycle, the
  ``/v1/conversations`` create/update/items mutations, the
  ``/v1/vector_stores`` create/update/attach/file_batches mutations,
  the stored-completion metadata update, the eval spec
  create/update, ``/v1/fine_tuning/jobs/{id}/resume`` (which
  re-admits a paused job onto the executor), and the entire
  ``/harness/keys`` mint/rotate/patch lifecycle all accepted
  new work mid-drain — a pod could mint keys, files, and stores in
  the same window its orchestrator believed it was refusing work.
  Each handler now calls ``_drain_refusal(metrics)`` at the
  admission point — after the idempotency replay short-circuit (a
  stored replay is a read and stays open) and before the store
  mutation — through the new module-level helper that ``_work_gate``
  also calls, so there is one canonical refusal.

Sealed ``drain_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from quant_fund.utils.reproducibility import git_revision

__all__ = ["drain_audit", "drain_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = secrets.token_urlsafe(32)
_WAIT_S = 15.0
_DRAIN = "/harness/drain"
_JOBS = "/harness/jobs"
_RUNS = "/harness/runs"
_EVALS = "/harness/evals"
_FILES = "/v1/files"
_UPLOADS = "/v1/uploads"
_CONVS = "/v1/conversations"
_VS = "/v1/vector_stores"
_KEYS = "/harness/keys"
_IDEM = "Idempotency-Key"
_CHAT = "/v1/chat/completions"
_RESPONSES = "/v1/responses"
_MESSAGES = "/v1/messages"
_LEGACY = "/v1/completions"
_TERMINAL = {"succeeded", "failed", "cancelled"}
_JSONL_LINE = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_BATCH_LINE = (
    b'{"custom_id":"d1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"fx1","messages":[{"role":"user","content":"x"}]}}\n'
)


# ---------------------------------------------------------------------------
# Stub plumbing — runners, app/client factories, poll helpers
# ---------------------------------------------------------------------------


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _make_app(
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    backend: Any | None = None,
    **create_kw: Any,
) -> FastAPI:
    """``create_app`` under the ambient env (``_audit_context`` already
    swept ``FX1_*``); ``backend`` installs one stub for every link."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415

    stub = backend if backend is not None else _StubBackend()
    return api_mod.create_app(
        harness=Harness(runner=runner or _fast_runner),
        backend_resolver=lambda *a, **k: stub,
        **create_kw,
    )


def _client(
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    api_key: str | None = _ROOT,
    **create_kw: Any,
) -> TestClient:
    """TestClient for a fresh app; ``api_key`` installs the root key."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

    if api_key is None:
        os.environ.pop(_API_KEY_ENV, None)
    else:
        os.environ[_API_KEY_ENV] = api_key
    app = _make_app(runner=runner, **create_kw)
    client = TestClient(app, raise_server_exceptions=False)
    # The app keeps a jobs executor pool; register its shutdown so probe
    # sections don't leak threads across the battery.
    stack = _RESOURCES.get(None)
    if stack is not None:
        stack.callback(app.state.jobs_executor.shutdown, False, cancel_futures=True)
    return client


def _h(auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    out = {"X-API-Key": auth} if auth else {}
    out.update(extra)
    return out


def _ih(key: str, auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    return _h(auth, **{_IDEM: key, **extra})


def _err_code(r: Any) -> str | None:
    """Flat ``code`` (harness grammar) or nested OpenAI ``error.code``."""
    if not r.headers.get("content-type", "").startswith("application/json"):
        return None
    body = r.json()
    code = body.get("code")
    if isinstance(code, str):
        return code
    err = body.get("error")
    if isinstance(err, dict) and isinstance(err.get("code"), str):
        return str(err["code"])
    return None


def _is_draining(r: Any) -> bool:
    return r.status_code == 503 and _err_code(r) == "draining"


def _anthropic_draining(r: Any) -> bool:
    """The Anthropic grammar carries ``type``, not ``code``."""
    if r.status_code != 503:
        return False
    body = r.json()
    return (
        body.get("type") == "error"
        and isinstance(body.get("error"), dict)
        and body["error"].get("type") == "api_error"
        and "draining" in str(body["error"].get("message", ""))
    )


def _mint(client: TestClient, root: str, **fields: Any) -> dict[str, Any]:
    r = client.post(_KEYS, json=fields, headers=_h(root))
    assert r.status_code == 201, f"key mint refused: {r.status_code} {r.text}"
    return dict(r.json())


def _key_card(client: TestClient, root: str, key_id: str) -> dict[str, Any]:
    return dict(client.get(f"{_KEYS}/{key_id}", headers=_h(root)).json())


def _upload_file(client: TestClient, *, purpose: str = "batch", line: bytes = _JSONL_LINE) -> str:
    r = client.post(
        _FILES,
        files={"file": ("a.jsonl", line)},
        data={"purpose": purpose},
        headers=_h(),
    )
    assert r.status_code == 200, f"file upload refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _make_upload(client: TestClient) -> str:
    r = client.post(
        _UPLOADS,
        json={
            "purpose": "batch",
            "filename": "in.jsonl",
            "bytes": len(_JSONL_LINE),
            "mime_type": "application/jsonl",
        },
        headers=_h(),
    )
    assert r.status_code == 200, f"upload create refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _make_eval_spec(client: TestClient) -> str:
    r = client.post(
        "/v1/evals",
        json={
            "name": "drain-spec",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 1, "backend": "byok"},
            },
        },
        headers=_h(),
    )
    assert r.status_code == 201, f"eval spec create refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _make_ft_job(client: TestClient, training_file: str) -> str:
    r = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": training_file},
        headers=_h(),
    )
    assert r.status_code == 200, f"ft submit refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _wait_job(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"{_JOBS}/{job_id}", headers=_h()).json()
        if st.get("status") in _TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _wait_response(client: TestClient, rid: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"{_RESPONSES}/{rid}", headers=_h()).json()
        if st.get("status") in {"completed", "failed", "cancelled", "incomplete"}:
            return st
        time.sleep(0.05)
    return st


def _wait_hits(sink: Any, n: int, timeout: float = _WAIT_S) -> None:
    end = time.monotonic() + timeout
    while len(sink.hits) < n and time.monotonic() < end:
        time.sleep(0.05)


@contextmanager
def _threaded(fn: Callable[[], Any]) -> Iterator[tuple[list[Any], list[BaseException]]]:
    """Run ``fn`` on a daemon thread; yields (results, errors), joined on
    exit — the deterministic shape for in-flight-at-drain probes."""
    results: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(fn())
        except BaseException as exc:  # noqa: BLE001 — probe captures every outcome
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield results, errors
    finally:
        thread.join(timeout=30)


# ---------------------------------------------------------------------------
# Probes: the latch — scope, body honesty, wait semantics, signals
# ---------------------------------------------------------------------------


def _latch_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()

    d = client.post(_DRAIN, headers=_h())
    body = d.json()
    out["drain_200_latches"] = d.status_code == 200 and body["draining"] is True
    out["drain_body_honest"] = (
        isinstance(body.get("inflight"), int)
        and isinstance(body.get("drained"), bool)
        and body["drained"] == (body["inflight"] == 0)
    )
    d2 = client.post(_DRAIN, headers=_h())
    out["drain_relatch_idempotent"] = d2.status_code == 200 and d2.json()["draining"] is True
    out["drain_ready_503"] = client.get("/ready", headers=_h()).status_code == 503
    out["drain_ready_code"] = _err_code(client.get("/ready", headers=_h())) == "draining"
    out["drain_health_reflects"] = client.get("/health").json().get("draining") is True
    out["drain_metrics_reflects"] = (
        client.get("/metrics", headers=_h()).json().get("draining") is True
    )
    prom = client.get("/metrics", params={"format": "prom"}, headers=_h())
    out["drain_prom_gauge"] = "fx1_draining 1" in prom.text
    out["drain_wait_drained_true"] = (
        client.post(_DRAIN, headers=_h(), params={"wait_s": 0.5}).json().get("drained") is True
    )
    # the latch is POST-only — there is no GET/DELETE drain surface and
    # no resume/undrain route anywhere in the grammar
    out["drain_get_405"] = client.get(_DRAIN, headers=_h()).status_code in (404, 405)
    out["no_resume_route"] = client.post("/harness/resume", headers=_h()).status_code in (
        404,
        405,
    )
    out["no_undrain_route"] = client.post("/harness/undrain", headers=_h()).status_code in (
        404,
        405,
    )
    return out


def _scope_probes() -> dict[str, bool]:
    """The latch is admin-only on every credential class."""
    out: dict[str, bool] = {}
    client = _client()

    read_key = _mint(client, _ROOT, scopes=["read"])
    write_key = _mint(client, _ROOT, scopes=["write"])
    admin_key = _mint(client, _ROOT, admin=True)

    r_read = client.post(_DRAIN, headers=_h(read_key["key"]))
    out["drain_read_scope_403"] = r_read.status_code == 403
    out["drain_read_scope_code"] = _err_code(r_read) == "insufficient_scope"
    r_write = client.post(_DRAIN, headers=_h(write_key["key"]))
    out["drain_write_scope_403"] = r_write.status_code == 403
    r_bad = client.post(_DRAIN, headers=_h("fx1k_forged"))
    out["drain_bad_key_401"] = r_bad.status_code == 401
    r_none = client.post(_DRAIN)
    out["drain_no_key_401"] = r_none.status_code == 401
    # the refused scope calls never armed the latch — a write-scoped
    # submit still reaches the gate rather than a 503 draining
    r_submit = client.post(_JOBS, json={"command": "doctor"}, headers=_h(write_key["key"]))
    out["drain_scope_refusals_did_not_latch"] = not _is_draining(r_submit)
    r_admin = client.post(_DRAIN, headers=_h(admin_key["key"]))
    out["drain_admin_200"] = r_admin.status_code == 200 and r_admin.json()["draining"] is True
    out["drain_admin_relatch_200"] = (
        client.post(_DRAIN, headers=_h(admin_key["key"])).status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# Probes: the refusal set — every mutating surface answers 503 draining
# ---------------------------------------------------------------------------


def _seed_store(client: TestClient) -> dict[str, Any]:
    """Pre-drain state the probes point at: one of every stored resource
    so each drain-time mutation is genuinely admissible."""
    file_id = _upload_file(client, purpose="batch", line=_BATCH_LINE)
    ft_file_id = _upload_file(client, purpose="fine-tune")
    conv = client.post(_CONVS, json={"metadata": {"seed": "1"}}, headers=_h())
    assert conv.status_code == 200, f"conv seed refused: {conv.status_code} {conv.text}"
    vs = client.post(_VS, json={"name": "drain-vs"}, headers=_h())
    assert vs.status_code == 200, f"vs seed refused: {vs.status_code} {vs.text}"
    upload_id = _make_upload(client)
    part = client.post(
        f"{_UPLOADS}/{upload_id}/parts", files={"data": ("p", _JSONL_LINE)}, headers=_h()
    )
    assert part.status_code == 200, f"part seed refused: {part.status_code} {part.text}"
    spec_id = _make_eval_spec(client)
    minted = _mint(client, _ROOT, name="drain-target")
    chat = client.post(
        _CHAT,
        json={"model": "fx1", "messages": [{"role": "user", "content": "seed"}]},
        headers=_h(),
    )
    assert chat.status_code == 200, f"chat seed refused: {chat.status_code} {chat.text}"
    ft_id = _make_ft_job(client, ft_file_id)
    return {
        "file_id": file_id,
        "ft_file_id": ft_file_id,
        "conv_id": str(conv.json()["id"]),
        "vs_id": str(vs.json()["id"]),
        "upload_id": upload_id,
        "part_id": part.json()["id"],
        "spec_id": spec_id,
        "minted": minted,
        "chat_id": chat.json()["id"],
        "ft_id": ft_id,
    }


def _refusal_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    seed = _seed_store(client)
    d = client.post(_DRAIN, headers=_h())
    assert d.status_code == 200

    # --- model-call surfaces (slot-gated) ---
    out["refuse_chat"] = _is_draining(
        client.post(
            _CHAT,
            json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
            headers=_h(),
        )
    )
    out["refuse_responses"] = _is_draining(
        client.post(_RESPONSES, json={"model": "fx1", "input": "x"}, headers=_h())
    )
    out["refuse_messages"] = _anthropic_draining(
        client.post(
            _MESSAGES,
            json={
                "model": "fx1",
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "x"}],
            },
            headers=_h(),
        )
    )
    out["refuse_completions"] = _is_draining(
        client.post(_LEGACY, json={"model": "fx1", "prompt": "x"}, headers=_h())
    )
    out["refuse_embeddings"] = _is_draining(
        client.post("/v1/embeddings", json={"model": "fx1", "input": "x"}, headers=_h())
    )
    # --- harness submit surfaces ---
    out["refuse_jobs"] = _is_draining(client.post(_JOBS, json={"command": "doctor"}, headers=_h()))
    out["refuse_runs"] = _is_draining(client.post(_RUNS, json={"command": "doctor"}, headers=_h()))
    out["refuse_evals"] = _is_draining(
        client.post(
            _EVALS,
            json={"suite": "tooluse", "backend": "byok", "seed": 0},
            headers=_h(),
        )
    )
    out["refuse_complete"] = _is_draining(
        client.post(
            "/harness/complete",
            json={"backend": "byok", "messages": [{"role": "user", "content": "x"}]},
            headers=_h(),
        )
    )
    out["refuse_complete_stream"] = _is_draining(
        client.post(
            "/harness/complete/stream",
            json={"backend": "byok", "messages": [{"role": "user", "content": "x"}]},
            headers=_h(),
        )
    )
    out["refuse_complete_batch"] = _is_draining(
        client.post(
            "/harness/complete/batch",
            json={
                "backend": "byok",
                "requests": [{"messages": [{"role": "user", "content": "x"}]}],
            },
            headers=_h(),
        )
    )
    out["refuse_backend_probe"] = _is_draining(
        client.post("/harness/backends/byok/probe", headers=_h())
    )
    # --- eval spec + run surfaces ---
    out["refuse_eval_spec_create"] = _is_draining(
        client.post(
            "/v1/evals",
            json={
                "name": "drain-spec-2",
                "data_source_config": {
                    "type": "custom",
                    "item_schema": {"suite": "tooluse", "seed": 2, "backend": "byok"},
                },
            },
            headers=_h(),
        )
    )
    out["refuse_eval_spec_update"] = _is_draining(
        client.post(f"/v1/evals/{seed['spec_id']}", json={"name": "renamed"}, headers=_h())
    )
    out["refuse_eval_run_create"] = _is_draining(
        client.post(f"/v1/evals/{seed['spec_id']}/runs", json={"model": "fx1"}, headers=_h())
    )
    # --- files + uploads lifecycle ---
    out["refuse_file_upload"] = _is_draining(
        client.post(
            _FILES,
            files={"file": ("b.jsonl", _JSONL_LINE)},
            data={"purpose": "batch"},
            headers=_h(),
        )
    )
    out["refuse_upload_create"] = _is_draining(
        client.post(
            _UPLOADS,
            json={
                "purpose": "batch",
                "filename": "late.jsonl",
                "bytes": len(_JSONL_LINE),
                "mime_type": "application/jsonl",
            },
            headers=_h(),
        )
    )
    out["refuse_upload_part"] = _is_draining(
        client.post(
            f"{_UPLOADS}/{seed['upload_id']}/parts",
            files={"data": ("p", _JSONL_LINE)},
            headers=_h(),
        )
    )
    out["refuse_upload_complete"] = _is_draining(
        client.post(
            f"{_UPLOADS}/{seed['upload_id']}/complete",
            json={"part_ids": [seed["part_id"]]},
            headers=_h(),
        )
    )
    # --- batches ---
    out["refuse_openai_batch"] = _is_draining(
        client.post(
            "/v1/batches",
            json={
                "input_file_id": seed["file_id"],
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
            headers=_h(),
        )
    )
    out["refuse_anthropic_batch"] = _anthropic_draining(
        client.post(
            "/v1/messages/batches",
            json={
                "requests": [
                    {
                        "custom_id": "d1",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 4,
                            "messages": [{"role": "user", "content": "x"}],
                        },
                    }
                ]
            },
            headers=_h(),
        )
    )
    # --- fine tuning ---
    out["refuse_ft_submit"] = _is_draining(
        client.post(
            "/v1/fine_tuning/jobs",
            json={"model": "fx1", "training_file": seed["ft_file_id"]},
            headers=_h(),
        )
    )
    out["refuse_ft_resume"] = _is_draining(
        client.post(f"/v1/fine_tuning/jobs/{seed['ft_id']}/resume", headers=_h())
    )
    # --- stored-resource mutations ---
    out["refuse_conv_create"] = _is_draining(
        client.post(_CONVS, json={"metadata": {"late": "1"}}, headers=_h())
    )
    out["refuse_conv_update"] = _is_draining(
        client.post(f"{_CONVS}/{seed['conv_id']}", json={"metadata": {"late": "2"}}, headers=_h())
    )
    out["refuse_conv_items_add"] = _is_draining(
        client.post(
            f"{_CONVS}/{seed['conv_id']}/items",
            json={
                "items": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "x"}],
                    }
                ]
            },
            headers=_h(),
        )
    )
    out["refuse_vs_create"] = _is_draining(client.post(_VS, json={"name": "late-vs"}, headers=_h()))
    out["refuse_vs_update"] = _is_draining(
        client.post(f"{_VS}/{seed['vs_id']}", json={"name": "renamed"}, headers=_h())
    )
    out["refuse_vs_attach"] = _is_draining(
        client.post(f"{_VS}/{seed['vs_id']}/files", json={"file_id": seed["file_id"]}, headers=_h())
    )
    out["refuse_vs_file_batch"] = _is_draining(
        client.post(
            f"{_VS}/{seed['vs_id']}/file_batches",
            json={"file_ids": [seed["file_id"]]},
            headers=_h(),
        )
    )
    out["refuse_chat_metadata_update"] = _is_draining(
        client.post(f"{_CHAT}/{seed['chat_id']}", json={"metadata": {"k": "v"}}, headers=_h())
    )
    # --- the key lifecycle ---
    out["refuse_key_mint"] = _is_draining(client.post(_KEYS, json={"name": "late"}, headers=_h()))
    out["refuse_key_rotate"] = _is_draining(
        client.post(f"{_KEYS}/{seed['minted']['id']}/rotate", json={}, headers=_h())
    )
    out["refuse_key_patch"] = _is_draining(
        client.patch(f"{_KEYS}/{seed['minted']['id']}", json={"name": "renamed"}, headers=_h())
    )
    revoked = client.delete(f"{_KEYS}/{seed['minted']['id']}", headers=_h())
    out["revoke_key_open"] = revoked.status_code == 200 and revoked.json().get("enabled") is False
    return out


# ---------------------------------------------------------------------------
# Probes: read-through — every read and control-plane surface stays open
# ---------------------------------------------------------------------------


def _read_through_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    own = _seed_store(client)
    client.post(_DRAIN, headers=_h())

    # --- ops reads ---
    out["read_health"] = client.get("/health").status_code == 200
    out["read_metrics"] = client.get("/metrics", headers=_h()).status_code == 200
    out["read_version"] = client.get("/harness/version", headers=_h()).status_code == 200
    out["read_capabilities"] = client.get("/harness/capabilities", headers=_h()).status_code == 200
    out["read_commands"] = client.get("/harness/commands", headers=_h()).status_code == 200
    out["read_self"] = client.get("/harness/self", headers=_h()).status_code == 200
    out["read_openapi"] = client.get("/openapi.json", headers=_h()).status_code == 200
    # --- stored-resource reads ---
    out["read_jobs_list"] = client.get(_JOBS, headers=_h()).status_code == 200
    out["read_evals_list"] = client.get(_EVALS, headers=_h()).status_code == 200
    out["read_eval_specs"] = client.get("/v1/evals", headers=_h()).status_code == 200
    out["read_eval_spec"] = (
        client.get(f"/v1/evals/{own['spec_id']}", headers=_h()).status_code == 200
    )
    out["read_files_list"] = client.get(_FILES, headers=_h()).status_code == 200
    out["read_file_get"] = client.get(f"{_FILES}/{own['file_id']}", headers=_h()).status_code == 200
    out["read_file_content"] = (
        client.get(f"{_FILES}/{own['file_id']}/content", headers=_h()).status_code == 200
    )
    # uploads carry no read surface at all — the lifecycle is
    # create/parts/complete/cancel; pin the absent routes honestly
    out["read_uploads_no_list_route"] = client.get(_UPLOADS, headers=_h()).status_code == 404
    out["read_uploads_no_get_route"] = (
        client.get(f"{_UPLOADS}/{own['upload_id']}", headers=_h()).status_code == 404
    )
    out["read_batches_list"] = client.get("/v1/batches", headers=_h()).status_code == 200
    out["read_abatches_list"] = client.get("/v1/messages/batches", headers=_h()).status_code == 200
    out["read_ft_list"] = client.get("/v1/fine_tuning/jobs", headers=_h()).status_code == 200
    out["read_ft_get"] = (
        client.get(f"/v1/fine_tuning/jobs/{own['ft_id']}", headers=_h()).status_code == 200
    )
    out["read_ft_events"] = (
        client.get(f"/v1/fine_tuning/jobs/{own['ft_id']}/events", headers=_h()).status_code == 200
    )
    out["read_conv_get"] = client.get(f"{_CONVS}/{own['conv_id']}", headers=_h()).status_code == 200
    out["read_conv_items"] = (
        client.get(f"{_CONVS}/{own['conv_id']}/items", headers=_h()).status_code == 200
    )
    out["read_vs_get"] = client.get(f"{_VS}/{own['vs_id']}", headers=_h()).status_code == 200
    out["read_vs_files"] = (
        client.get(f"{_VS}/{own['vs_id']}/files", headers=_h()).status_code == 200
    )
    out["read_chat_get"] = client.get(f"{_CHAT}/{own['chat_id']}", headers=_h()).status_code == 200
    out["read_keys_list"] = client.get(_KEYS, headers=_h()).status_code == 200
    out["read_key_get"] = (
        client.get(f"{_KEYS}/{own['minted']['id']}", headers=_h()).status_code == 200
    )
    out["read_key_usage"] = (
        client.get(f"{_KEYS}/{own['minted']['id']}/usage", headers=_h()).status_code == 200
    )
    out["read_models"] = client.get("/v1/models", headers=_h()).status_code == 200
    # --- cancels + pauses + deletes: control-plane teardown stays open ---
    out["cancel_upload_open"] = (
        client.post(f"{_UPLOADS}/{own['upload_id']}/cancel", headers=_h()).status_code == 200
    )
    out["cancel_eval_open"] = (
        client.delete(f"{_EVALS}/eval_missing", headers=_h()).status_code == 404
    )
    # ft pause/cancel answer their state-machine verdict (200/409),
    # never a drain refusal — teardown stays open mid-drain
    p = client.post(f"/v1/fine_tuning/jobs/{own['ft_id']}/pause", headers=_h())
    out["pause_ft_open"] = p.status_code in (200, 409) and p.status_code != 503
    c = client.post(f"/v1/fine_tuning/jobs/{own['ft_id']}/cancel", headers=_h())
    out["cancel_ft_open"] = c.status_code in (200, 409) and c.status_code != 503
    out["delete_conv_open"] = (
        client.delete(f"{_CONVS}/{own['conv_id']}", headers=_h()).status_code == 200
    )
    out["delete_vs_open"] = client.delete(f"{_VS}/{own['vs_id']}", headers=_h()).status_code == 200
    out["delete_file_open"] = (
        client.delete(f"{_FILES}/{own['file_id']}", headers=_h()).status_code == 200
    )
    out["delete_chat_open"] = (
        client.delete(f"{_CHAT}/{own['chat_id']}", headers=_h()).status_code == 200
    )
    out["delete_eval_spec_open"] = (
        client.delete(f"/v1/evals/{own['spec_id']}", headers=_h()).status_code == 200
    )
    out["delete_model_not_drain_refused"] = (
        client.delete("/v1/models/ft:never", headers=_h()).status_code != 503
    )
    # --- advisory + read-verbs-dressed-as-POST stay open ---
    out["verify_stays_live"] = (
        client.post("/receipts/verify", json={"receipt": {}}, headers=_h()).status_code != 503
    )
    out["gate_check_advisory"] = (
        client.post("/harness/gate/check", json={"text": "probe"}, headers=_h()).status_code == 200
    )
    out["score_advisory"] = (
        client.post("/harness/score", json={"input": "probe"}, headers=_h()).status_code == 200
    )
    out["moderations_advisory"] = (
        client.post("/v1/moderations", json={"input": "x"}, headers=_h()).status_code == 200
    )
    ct = client.post(
        "/v1/messages/count_tokens",
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        headers=_h(),
    )
    out["count_tokens_not_drain_refused"] = ct.status_code != 503
    vs_search = client.post(
        "/v1/vector_stores/vs_missing/search", json={"query": "x"}, headers=_h()
    )
    out["vs_search_read_not_drain"] = vs_search.status_code != 503
    return out


# ---------------------------------------------------------------------------
# Probes: in-flight honesty — admitted work completes past the latch
# ---------------------------------------------------------------------------


def _inflight_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.jobs_audit import _Runner  # noqa: PLC0415

    # --- a bg job admitted pre-drain finishes post-latch ---
    gate = threading.Event()
    runner = _Runner(gate=gate)
    client = _client(runner=runner)
    jid = client.post(_JOBS, json={"command": "doctor"}, headers=_h()).json()["job_id"]
    end = time.monotonic() + _WAIT_S
    while not runner.entered.is_set() and time.monotonic() < end:
        time.sleep(0.02)
    out["inflight_job_started"] = runner.entered.is_set()
    d = client.post(_DRAIN, headers=_h(), params={"wait_s": 0.05})
    out["inflight_drain_reports_inflight"] = (
        d.status_code == 200 and d.json()["inflight"] >= 1 and d.json()["drained"] is False
    )
    gate.set()
    rec = _wait_job(client, jid)
    out["inflight_job_completes"] = rec.get("status") == "succeeded"
    out["inflight_job_record_honest"] = (
        rec.get("result", {}).get("exit_code") == 0 and rec.get("finished_at") is not None
    )
    d2 = client.post(_DRAIN, headers=_h(), params={"wait_s": 1.0})
    out["drain_then_drained_true"] = d2.json().get("drained") is True
    out["inflight_zero_after"] = d2.json().get("inflight") == 0

    # --- a sync /harness/runs request in-flight at latch completes honestly ---
    gate2 = threading.Event()
    runner2 = _Runner(gate=gate2)
    client2 = _client(runner=runner2)
    with _threaded(lambda: client2.post(_RUNS, json={"command": "doctor"}, headers=_h())) as (
        results,
        errors,
    ):
        end = time.monotonic() + _WAIT_S
        while not runner2.entered.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        out["inflight_sync_entered"] = runner2.entered.is_set()
        client2.post(_DRAIN, headers=_h())
        gate2.set()
    out["inflight_sync_completes_200"] = bool(
        not errors and results and results[0].status_code == 200 and results[0].json()["ok"]
    )

    # --- a gated model call in-flight completes past the latch ---
    from fx1.serve.conv_audit import _GateBackend  # noqa: PLC0415

    gb = _GateBackend()
    client3 = _client(backend=gb)
    with _threaded(
        lambda: client3.post(
            _CHAT,
            json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
            headers=_h(),
        )
    ) as (results3, errors3):
        end = time.monotonic() + _WAIT_S
        while not gb.entered.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        out["inflight_model_entered"] = gb.entered.is_set()
        client3.post(_DRAIN, headers=_h())
        gb.gate.set()
    out["inflight_model_completes_200"] = bool(
        not errors3 and results3 and results3[0].status_code == 200
    )

    # --- a background response admitted pre-drain completes post-latch ---
    gb2 = _GateBackend()
    client4 = _client(backend=gb2)
    bg = client4.post(
        _RESPONSES,
        json={"model": "fx1", "input": "bg", "background": True},
        headers=_h(),
    )
    rid = bg.json()["id"]
    end = time.monotonic() + _WAIT_S
    while not gb2.entered.is_set() and time.monotonic() < end:
        time.sleep(0.02)
    client4.post(_DRAIN, headers=_h())
    gb2.gate.set()
    rec4 = _wait_response(client4, rid)
    out["inflight_bg_response_completes"] = rec4.get("status") == "completed"

    # --- a streaming request admitted pre-drain emits every frame ---
    gb3 = _GateBackend()
    client5 = _client(backend=gb3)
    with _threaded(
        lambda: client5.post(
            _CHAT,
            json={
                "model": "fx1",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
            headers=_h(),
        )
    ) as (results5, errors5):
        end = time.monotonic() + _WAIT_S
        while not gb3.entered.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        out["inflight_stream_entered"] = gb3.entered.is_set()
        client5.post(_DRAIN, headers=_h())
        gb3.gate.set()
    ok5 = bool(not errors5 and results5)
    out["inflight_stream_200"] = ok5 and results5[0].status_code == 200
    body5 = results5[0].text if ok5 else ""
    out["inflight_stream_frames_complete"] = "data:" in body5 and "[DONE]" in body5
    return out


# ---------------------------------------------------------------------------
# Probes: the submit-then-drain race boundary + idempotency
# ---------------------------------------------------------------------------


def _race_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.jobs_audit import _Runner  # noqa: PLC0415

    gate = threading.Event()
    runner = _Runner(gate=gate)
    client = _client(runner=runner)

    # submit immediately before the latch — admitted, runs to terminal
    pre = client.post(_JOBS, json={"command": "doctor"}, headers=_h())
    out["race_pre_drain_accepted"] = pre.status_code == 202
    jid = pre.json()["job_id"]
    client.post(_DRAIN, headers=_h())
    gate.set()
    rec = _wait_job(client, jid)
    out["race_pre_drain_completes"] = rec.get("status") == "succeeded"
    # submit immediately after the latch — refused
    post = client.post(_JOBS, json={"command": "doctor"}, headers=_h())
    out["race_post_drain_refused"] = _is_draining(post)
    # the boundary is clean: the refused submit minted no ghost record —
    # the store still lists exactly the pre-drain job
    listing = client.get(_JOBS, headers=_h()).json()
    jobs = [j for j in listing.get("jobs", [])]
    out["race_no_ghost_record"] = len(jobs) == 1 and jobs[0].get("job_id") == jid
    return out


def _idem_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    state_dir = _temporary_directory() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # a keyed submit stored pre-drain replays under drain — replays are reads
    client = _client(state_dir=state_dir)
    first = client.post(_JOBS, json={"command": "doctor"}, headers=_ih("dk-1"))
    out["idem_seed_202"] = first.status_code == 202
    jid = first.json()["job_id"]
    _wait_job(client, jid)
    client.post(_DRAIN, headers=_h())
    replay = client.post(_JOBS, json={"command": "doctor"}, headers=_ih("dk-1"))
    out["idem_replay_serves_under_drain"] = (
        replay.status_code == 202 and replay.json().get("job_id") == jid
    )
    # a conflicting keyed body under drain still answers 409 — the
    # stored-fingerprint verdict is a read, not work
    conflict = client.post(
        _JOBS,
        json={"command": "doctor", "extra_args": ["x"]},
        headers=_ih("dk-1"),
    )
    out["idem_conflict_under_drain"] = conflict.status_code == 409
    # a keyed submit refused under drain stores nothing — the claim
    # never becomes a phantom record a later retry could replay
    refused = client.post(_JOBS, json={"command": "doctor"}, headers=_ih("dk-2"))
    out["idem_keyed_refused"] = _is_draining(refused)
    client2 = _client(state_dir=state_dir)  # restart clears the latch
    fresh = client2.post(_JOBS, json={"command": "doctor"}, headers=_ih("dk-2"))
    out["idem_no_phantom_claim"] = (
        fresh.status_code == 202 and fresh.json().get("replayed") is not True
    )
    return out


# ---------------------------------------------------------------------------
# Probes: recovery — no resume; restart is the un-latch
# ---------------------------------------------------------------------------


def _recovery_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    state_dir = _temporary_directory() / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    c1 = _client(state_dir=state_dir)
    c1.post(_JOBS, json={"command": "doctor"}, headers=_h())
    c1.post(_DRAIN, headers=_h())
    out["drained_refuses"] = _is_draining(c1.post(_JOBS, json={"command": "doctor"}, headers=_h()))
    # the latch is process-local: a restarted app on the same state dir
    # accepts work again — drain state deliberately does not journal
    c2 = _client(state_dir=state_dir)
    r = c2.post(_JOBS, json={"command": "doctor"}, headers=_h())
    out["restart_clears_latch"] = r.status_code == 202
    out["restart_health_clear"] = c2.get("/health").json().get("draining") is False
    out["restart_ready_200"] = c2.get("/ready", headers=_h()).status_code == 200
    # and the restarted app re-arms on the next drain
    d2 = c2.post(_DRAIN, headers=_h())
    out["rearm_after_restart"] = d2.status_code == 200 and d2.json()["draining"] is True
    out["rearm_refuses_again"] = _is_draining(
        c2.post(_JOBS, json={"command": "doctor"}, headers=_h())
    )
    # drain-then-restart leaves stored resources intact — reads restore
    out["restart_state_intact"] = c2.get(_FILES, headers=_h()).status_code == 200
    return out


# ---------------------------------------------------------------------------
# Probes: drain-time webhooks + keys + metering + cascades + envelope
# ---------------------------------------------------------------------------


def _webhook_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    # Synthetic loopback receiver: opt in narrowly inside the swept audit context.
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415
    from fx1.serve.jobs_audit import _Runner  # noqa: PLC0415
    from fx1.serve.webhook_audit import _Sink  # noqa: PLC0415
    from fx1.serve.webhooks import (  # noqa: PLC0415
        WEBHOOK_SIGNATURE_HEADER,
        WEBHOOK_TIMESTAMP_HEADER,
        verify_webhook,
    )

    sink = _Sink()
    stack = _RESOURCES.get(None)
    if stack is not None:
        stack.callback(sink.close)

    secret = "drain-hook-secret"
    gate = threading.Event()
    runner = _Runner(gate=gate)
    client = _client(runner=runner)

    n0 = len(sink.hits)
    jid = client.post(
        _JOBS,
        json={
            "command": "doctor",
            "callback_url": sink.url("/hook"),
            "callback_secret": secret,
        },
        headers=_h(),
    ).json()["job_id"]
    end = time.monotonic() + _WAIT_S
    while not runner.entered.is_set() and time.monotonic() < end:
        time.sleep(0.02)
    # latch mid-flight — the completing job still fires its signed webhook
    client.post(_DRAIN, headers=_h())
    gate.set()
    rec = _wait_job(client, jid)
    out["webhook_job_terminal"] = rec.get("status") == "succeeded"
    _wait_hits(sink, n0 + 1)
    hit = sink.hits[n0] if len(sink.hits) > n0 else None
    out["webhook_fires_past_drain"] = hit is not None
    sig = hit.headers.get(WEBHOOK_SIGNATURE_HEADER) if hit is not None else None
    ts = hit.headers.get(WEBHOOK_TIMESTAMP_HEADER) if hit is not None else None
    out["webhook_hmac_verifies"] = (
        hit is not None
        and verify_webhook(secret, ts, sig, hit.body)
        and isinstance(sig, str)
        and sig.startswith("sha256=")
    )
    out["webhook_payload_terminal"] = hit is not None and b"succeeded" in hit.body
    return out


def _keys_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    write_key = _mint(client, _ROOT, scopes=["read", "write"])
    admin_key = _mint(client, _ROOT, admin=True)

    client.post(_DRAIN, headers=_h())
    # drain revokes nothing: existing keys still authenticate their
    # standing scopes — reads answer, writes hit the drain gate
    out["existing_key_read_200"] = (
        client.get(_JOBS, headers=_h(write_key["key"])).status_code == 200
    )
    r = client.post(_JOBS, json={"command": "doctor"}, headers=_h(write_key["key"]))
    out["existing_key_write_drains"] = _is_draining(r)
    out["admin_reads_survive"] = client.get(_KEYS, headers=_h(admin_key["key"])).status_code == 200
    # the drain call itself still works for any admin credential
    out["admin_key_relatch_200"] = (
        client.post(_DRAIN, headers=_h(admin_key["key"])).status_code == 200
    )
    # key records untouched by the latch — the minted target still exists
    rec = client.get(f"{_KEYS}/{write_key['id']}", headers=_h())
    out["key_record_untouched"] = rec.status_code == 200 and rec.json().get("enabled") is not False
    return out


def _metering_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    minted = _mint(client, _ROOT)
    raw, kid = minted["key"], minted["id"]
    admin_key = _mint(client, _ROOT, admin=True)
    ro = _mint(client, _ROOT, scopes=["read"])

    uses0 = _key_card(client, _ROOT, kid)["uses"]
    client.post(_DRAIN, headers=_h())
    # pinned actual: a drain-refused submit still bills a use — the
    # counter moves at authenticate, before the gate verdict
    client.post(_JOBS, json={"command": "doctor"}, headers=_h(raw))
    uses1 = _key_card(client, _ROOT, kid)["uses"]
    out["drain_refused_bills_use"] = uses1 - uses0 == 1
    # the drain POST itself is an authenticated call — it bills too
    a0 = _key_card(client, _ROOT, admin_key["id"])["uses"]
    client.post(_DRAIN, headers=_h(admin_key["key"]))
    a1 = _key_card(client, _ROOT, admin_key["id"])["uses"]
    out["drain_call_bills_use"] = a1 - a0 == 1
    # scope refusals still bill nothing — auth-layer verdicts precede
    ro0 = _key_card(client, _ROOT, ro["id"])["uses"]
    client.post(_JOBS, json={"command": "doctor"}, headers=_h(ro["key"]))
    out["scope_refusal_still_no_bill"] = _key_card(client, _ROOT, ro["id"])["uses"] == ro0
    # no model work ran — tokens stay unbilled through the refusals
    out["drain_tokens_never_billed"] = _key_card(client, _ROOT, kid)["tokens_used"] == 0
    return out


def _envelope_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    client.post(_DRAIN, headers=_h())

    r_h = client.post(_JOBS, json={"command": "doctor"}, headers=_h())
    body_h = r_h.json()
    out["envelope_harness_shape"] = (
        "detail" in body_h and body_h.get("code") == "draining" and "error" not in body_h
    )
    r_v = client.post(
        _CHAT,
        json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
        headers=_h(),
    )
    err = r_v.json().get("error", {})
    out["envelope_openai_shape"] = (
        err.get("code") == "draining"
        and err.get("type") == "service_unavailable"
        and "draining" in str(err.get("message", ""))
    )
    r_a = client.post(
        _MESSAGES,
        json={
            "model": "fx1",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "x"}],
        },
        headers=_h(),
    )
    body_a = r_a.json()
    out["envelope_anthropic_shape"] = (
        body_a.get("type") == "error"
        and body_a.get("error", {}).get("type") == "api_error"
        and "draining" in str(body_a.get("error", {}).get("message", ""))
    )
    out["envelope_503_uniform"] = r_h.status_code == r_v.status_code == r_a.status_code == 503
    out["envelope_message_uniform"] = all(
        "draining" in json.dumps(r.json()) for r in (r_h, r_v, r_a)
    )
    return out


def _cascade_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.jobs_audit import _Runner  # noqa: PLC0415

    runner = _Runner()
    client = _client(runner=runner)
    fid = _upload_file(client, line=_BATCH_LINE)
    client.post(_DRAIN, headers=_h())

    # /harness/jobs/batch under drain: every item refused, none executed
    b = client.post(
        f"{_JOBS}/batch",
        json={"jobs": [{"command": "doctor"}, {"command": "doctor"}]},
        headers=_h(),
    )
    body = b.json()
    items = body.get("jobs") or []
    out["cascade_batch_itemwise_drain"] = (
        b.status_code == 202
        and len(items) == 2
        and all(item.get("code") == "draining" for item in items)
    )
    out["cascade_batch_counts"] = body.get("submitted") == 0 and body.get("failed") == 2
    out["cascade_batch_zero_exec"] = len(runner.calls) == 0

    # /v1/batches refused at submit — the input file's lines never ran
    r = client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers=_h(),
    )
    out["cascade_openai_batch_refused"] = _is_draining(r)
    out["cascade_openai_batch_no_partial"] = (
        client.get("/v1/batches", headers=_h()).json().get("data", []) == []
    )
    # a refused spec mint leaves no record — the list keeps the seeded set
    n0 = len(client.get("/v1/evals", headers=_h()).json().get("data", []))
    client.post(
        "/v1/evals",
        json={
            "name": "ghost",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 9, "backend": "byok"},
            },
        },
        headers=_h(),
    )
    n1 = len(client.get("/v1/evals", headers=_h()).json().get("data", []))
    out["cascade_refused_mint_no_record"] = n1 == n0
    return out


# ---------------------------------------------------------------------------
# Aggregator + sealed bench
# ---------------------------------------------------------------------------


def drain_audit() -> dict[str, Any]:
    """Run the drain-lifecycle battery; returns literal bools."""
    from fx1.serve.conv_audit import _audit_context  # noqa: PLC0415

    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_latch_probes())
        out.update(_scope_probes())
        out.update(_refusal_probes())
        out.update(_read_through_probes())
        out.update(_inflight_probes())
        out.update(_race_probes())
        out.update(_idem_probes())
        out.update(_recovery_probes())
        out.update(_webhook_probes())
        out.update(_keys_probes())
        out.update(_metering_probes())
        out.update(_envelope_probes())
        out.update(_cascade_probes())
        return out


def drain_audit_bench(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal drain-audit results; run the battery when results are omitted."""
    r = drain_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "drain_audit",
        "schema": "drain_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": [
                "harness.drain",
                "harness.jobs.submit",
                "harness.jobs.batch",
                "harness.runs",
                "harness.evals",
                "harness.complete",
                "harness.complete.stream",
                "harness.complete.batch",
                "harness.backends.probe",
                "harness.keys.mint",
                "harness.keys.rotate",
                "harness.keys.patch",
                "harness.keys.revoke",
                "openai.chat",
                "openai.responses",
                "openai.completions",
                "openai.embeddings",
                "openai.files",
                "openai.uploads",
                "openai.batches",
                "openai.evals",
                "openai.fine_tuning",
                "openai.conversations",
                "openai.vector_stores",
                "anthropic.messages",
                "anthropic.batches",
            ],
            "not_verified": [
                "real multi-process drain + kill (the latch is pinned process-local)",
                "real-socket HarnessClient drain leg (e2e_audit owns it)",
                "signal-driven drain",
            ],
            "not_executed": [
                "drain across a rolling deploy with live peers",
            ],
        },
        "interpretation": (
            "Every probe True means: on this checkout POST /harness/drain "
            "is a one-way admin latch whose body honestly reports "
            "draining/inflight/drained (wait_s bounded), /ready 503s for "
            "deregistration while /health + /metrics keep answering, "
            "every mutating surface — the slot-gated model calls, the "
            "jobs/evals/batches submits, the stored-resource writes "
            "(files, uploads, conversations, vector stores, eval specs, "
            "stored-completion metadata, ft resume), plus key mint/rotate/"
            "patch — refuse 503 draining in their own error grammar; key "
            "revocation remains open so compromised authority can be removed, "
            "every read plus cancels/deletes/pauses and the documented "
            "advisory preflights stay open, work admitted before the "
            "latch completes honestly (jobs, sync runs, model calls, "
            "background responses, SSE streams all finish past the "
            "latch), the submit boundary is clean with no ghost records, "
            "a pending signed webhook still fires once, existing keys "
            "still authenticate while refusing nothing else, a "
            "drain-refused call still bills uses (authenticate precedes "
            "the gate) while scope refusals bill nothing, a keyed refusal "
            "stores no phantom claim while stored replays keep serving, "
            "and the only recovery is restart — the latch is "
            "process-local and re-arms cleanly. SYNTHETIC stub "
            "backends/runners only — no research claim."
        ),
    }
    canon = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return out


if __name__ == "__main__":
    print(json.dumps(drain_audit_bench(), indent=1))
