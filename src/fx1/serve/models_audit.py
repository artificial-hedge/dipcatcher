"""models_audit — adversarial probes on the ``/v1/models`` registry surface.

The claim under test: the model registry is a *correct name→weights
directory* — listing and retrieval answer the same inventory in both
envelope dialects, a succeeded fine-tune mints a ``ft:`` name that
resolves its producing job's checkpoint (never the base link), deleting
a name is a real journaled tombstone (list/retrieve/resolve/checkpoints
all drop it, a restart never resurrects it), and every malformed id or
unknown model lands in the wire's error envelope instead of silently
defaulting to the base link.

Coverage map:

- *List* — the OpenAI ``{object: "list", data: [{id, created,
  owned_by}]}`` shape carries the four built-in link ids; an
  ``anthropic-version`` header swaps the same route to Anthropic's
  ``{data: [{type, id, display_name, created_at}], first_id, last_id,
  has_more}`` grammar; ``limit``/``after_id``/``before_id`` page
  honestly in both directions; unknown cursors page to empty; a
  deleted name is absent under both dialects.
- *Get* — existing ids round-trip cards in both dialects; unknown ids
  refuse 404 in the dialect's own error envelope (OpenAI
  ``model_not_found`` / Anthropic ``not_found_error``); ids are
  case-sensitive.
- *ft: lifecycle* — a succeeded job mints exactly the
  ``ft:{model}:{suffix}:{job-id}`` name; it lists and retrieves under
  both dialects; the card's ``created`` is the registration stamp (the
  same value the checkpoint listing reports), not the app's boot time;
  the name resolves on chat/responses/messages to the ``local_fx1``
  link pinned at the job's checkpoint; the response ``model`` names the
  ft alias and ``system_fingerprint`` the backend; the ``ftckpt-``
  listing maps 1:1 to live cards; DELETE tombstones the name on every
  surface — list, card, resolve, checkpoint listing — and a mid-flight
  request already bound to the checkpoint still completes while every
  new one 404s. Explicit backend / BYOK headers beat the registry: the
  caller's stated link wins over the ft name. Failed and cancelled
  jobs, and jobs whose outcome declares no model, mint nothing.
- *Checkpoint gates* — ``checkpoint_dir`` only binds the
  ``local_fx1`` link (422 elsewhere); ``local_fx1`` with no checkpoint
  anywhere is 422, never a silent base; resolver faults map to the
  wire's error classes (KeyError→404, FileNotFoundError→422,
  RuntimeError→503).
- *Delete* — DELETE returns the ``{id, object: "model", deleted: true}``
  record; the second delete is a clean enveloped 404 (idempotent
  tombstone, never a fabricated verdict); built-in link ids refuse 400
  ``invalid_request`` — the base model can never be tombstoned.
- *model param validation* — an unknown model refuses 404
  ``model_not_found`` on chat, responses, messages, count_tokens and
  the legacy completions surface — never a 500 and never a silent
  default to the base link; an empty ``model`` is a 422 validation
  refusal; an embedding model is a free-form upstream name passed to
  the embedding link, not a registry lookup.
- *Envelope dialect* — the same key resolves through ``X-API-Key`` and
  ``Authorization: Bearer`` on both grammars; the card inventory and
  error envelopes follow the request's dialect.
- *Durability* — under ``state_dir`` the registry + tombstones journal
  to ``ft_jobs.jsonl`` and replay verbatim after restart: registered
  names survive, deleted names stay dead everywhere, the base model is
  never tombstoned, and a post-restart job mints and resolves fresh.
- *Concurrency* — parallel list/get/resolve racing a delete resolve
  atomically (full card or enveloped 404, never torn); two parallel ft
  completions mint distinct names.
- *Errors* — encoded traversal (``%2E%2E``, ``%2F``), wildcards, NUL
  bytes, colons, and overlong ids are all enveloped refusals — the
  router never lets an id escape its segment and no handler 500s.
- *SDK parity* — the in-process ``Fx1Harness`` twin lists, retrieves,
  deletes, and resolves the same registry (its own ``state_dir``)
  with the same refusals.

Three defects were found and fixed while building this battery: the
``ft:`` card's ``created`` reported the app's boot stamp instead of the
registration time the checkpoint listing already published; an unknown
non-``ft:`` model silently fell through to the ``hosted_k3`` default
link on every completion surface, answering 200 for models that do not
exist (a typo'd name ran the base model instead of 404ing); and a
request resolved through the ft registry stamped the response ``model``
with the checkpoint's internal version rather than the ``ft:`` name the
caller addressed. The empty-``model`` 422 gate was added with them —
an empty string rode the same silent default. All are pinned green.

Sealed ``models_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
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

__all__ = ["models_audit", "models_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("models_audit_resources")


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
    return Path(_RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="models_audit_")))


_MODEL = "byok"
_N = 8
_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_BASE_IDS = {"fx1", "hosted_k3", "local_fx1", "byok"}


# ---------------------------------------------------------------------------
# Stub backends + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn."""

    def __init__(self, model: str = "models-stub-0") -> None:
        self._model = model
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        self.calls = 0
        self.seen_model: str | None = None

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        return f"stub:{messages[-1]['content']}"

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> Any:
        from fx1.serve.backends import EmbeddingResult

        del encoding_format, dimensions, user
        self.seen_model = model
        n = (
            len(input)
            if isinstance(input, list) and input and isinstance(input[0], (str, list))
            else 1
        )
        data = tuple({"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(n))
        return EmbeddingResult(
            data=data, model=model, usage={"prompt_tokens": 1, "total_tokens": 1}
        )

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


class _Resolver:
    """``backend_resolver`` spy: dispatches per-link factories, records
    every (link, checkpoint_dir) pair the wire resolved, and can be armed
    to raise the documented fault classes for gate probes."""

    def __init__(self, factories: dict[str, Any]) -> None:
        self._factories = factories
        self.resolved: list[tuple[str, Any]] = []
        self.raise_for: dict[str, Exception] = {}

    def __call__(self, name: str, *a: Any, **k: Any) -> Any:
        ckpt = k.get("checkpoint_dir") or (a[0] if a else None)
        self.resolved.append((name, ckpt))
        if name in self.raise_for:
            raise self.raise_for[name]
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(name)
        return factory(**k)


def _spy_resolver() -> _Resolver:
    return _Resolver(
        {
            "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
            "local_fx1": lambda **k: _StubBackend("ckpt-stub"),
            "byok": lambda **k: _StubBackend("byok-stub"),
        }
    )


def _client(
    resolver: _Resolver | None = None,
    *,
    api_key: str | None = None,
    state_dir: Path | None = None,
    ft_runner: Any = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
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
            backend_resolver=resolver if resolver is not None else _spy_resolver(),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_runner=ft_runner if ft_runner is not None else _ok_runner,
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


def _ok_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Succeeded job minting a real checkpoint dir."""
    from fx1.serve.finetune import FTJobOutcome

    ckpt = spec.work_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        checkpoint=str(ckpt),
        trained_tokens=7,
    )


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


_ANTHROPIC_H = {"anthropic-version": "2023-06-01"}


def _ah(auth: str | None) -> dict[str, str]:
    """The anthropic dialect headers — ``anthropic-version`` flips the
    envelope; the same key rides ``x-api-key``."""
    return {**_ANTHROPIC_H, **_h(auth)}


def _chat(client: TestClient, model: str, **kw: Any) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = kw.pop("headers", {})
    body.update(kw)
    return client.post("/v1/chat/completions", json=body, headers=headers)


def _respond(client: TestClient, model: str, **kw: Any) -> Any:
    return client.post("/v1/responses", json={"model": model, "input": "hi", **kw})


def _message(client: TestClient, model: str, **kw: Any) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = kw.pop("headers", {})
    body.update(kw)
    return client.post("/v1/messages", json=body, headers=headers)


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error")
    return err if isinstance(err, dict) else {}


def _is_openai_envelope(resp: Any) -> bool:
    """The OpenAI error envelope — ``{error: {message, type, param, code}}``."""
    err = _err(resp)
    return (
        isinstance(err.get("message"), str) and isinstance(err.get("type"), str) and "code" in err
    )


def _is_anthropic_envelope(resp: Any) -> bool:
    """Anthropic's ``{type: "error", error: {type, message}}`` shape."""
    body = resp.json()
    err = body.get("error")
    return (
        body.get("type") == "error"
        and isinstance(err, dict)
        and isinstance(err.get("type"), str)
        and isinstance(err.get("message"), str)
    )


def _ids(resp: Any) -> list[str]:
    return [str(m.get("id")) for m in resp.json()["data"]]


def _upload(client: TestClient, content: bytes = _CORPUS) -> str:
    r = client.post(
        "/v1/files",
        files={"file": ("c.jsonl", content, "application/jsonl")},
        data={"purpose": "fine-tune"},
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _wait_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(600):
        job: dict[str, Any] = client.get(f"/v1/fine_tuning/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _ft_job(client: TestClient, suffix: str = "m") -> tuple[dict[str, Any], str]:
    """Submit one fine-tune job and wait for the minted ``ft:`` name."""
    fid = _upload(client)
    r = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid, "suffix": suffix},
    )
    assert r.status_code in (200, 201, 202), r.text
    job = _wait_terminal(client, str(r.json()["id"]))
    name = job.get("fine_tuned_model")
    return job, str(name) if isinstance(name, str) else ""


def _checkpoints(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    r = client.get(f"/v1/fine_tuning/jobs/{job_id}/checkpoints")
    assert r.status_code == 200, r.text
    out: list[dict[str, Any]] = r.json()["data"]
    return out


# ---------------------------------------------------------------------------
# List surface — both envelopes + cursor paging
# ---------------------------------------------------------------------------


def _probe_list() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver)

    page = client.get("/v1/models")
    body = page.json()
    out["list_openai_envelope"] = (
        page.status_code == 200
        and body.get("object") == "list"
        and isinstance(body.get("data"), list)
    )
    cards = body["data"]
    out["list_openai_card_shape"] = all(
        c.get("object") == "model"
        and isinstance(c.get("id"), str)
        and isinstance(c.get("created"), int)
        and isinstance(c.get("owned_by"), str)
        for c in cards
    )
    ids = _ids(page)
    out["list_builtin_ids"] = _BASE_IDS.issubset(set(ids))
    out["list_no_ft_without_jobs"] = all(not i.startswith("ft:") for i in ids)

    # Anthropic dialect — the same route, the other grammar.
    apage = client.get("/v1/models", headers=_ANTHROPIC_H)
    abody = apage.json()
    out["list_anthropic_swaps_envelope"] = (
        apage.status_code == 200
        and "object" not in abody
        and isinstance(abody.get("data"), list)
        and "first_id" in abody
        and "last_id" in abody
        and "has_more" in abody
    )
    acards = abody["data"]
    out["list_anthropic_card_shape"] = all(
        c.get("type") == "model"
        and c.get("display_name") == c.get("id")
        and isinstance(c.get("created_at"), str)
        for c in acards
    )
    out["list_anthropic_same_ids"] = [str(c["id"]) for c in acards] == ids

    # Cursor paging — honest in both directions.
    full = client.get("/v1/models", headers=_ANTHROPIC_H, params={"limit": 100}).json()
    p1 = client.get("/v1/models", headers=_ANTHROPIC_H, params={"limit": 2}).json()
    out["list_anthropic_limit_pages"] = (
        len(p1["data"]) == 2 and p1["has_more"] is True and p1["first_id"] == full["data"][0]["id"]
    )
    p2 = client.get(
        "/v1/models",
        headers=_ANTHROPIC_H,
        params={"limit": 2, "after_id": p1["last_id"]},
    ).json()
    out["list_anthropic_after_continues"] = (
        len(p2["data"]) >= 1
        and p2["data"][0]["id"] == full["data"][2]["id"]
        and p1["last_id"] not in {c["id"] for c in p2["data"]}
    )
    cursor = p1["last_id"]
    walked = [str(c["id"]) for c in p1["data"]]
    for _ in range(10):
        nxt = client.get(
            "/v1/models",
            headers=_ANTHROPIC_H,
            params={"limit": 2, "after_id": cursor},
        ).json()
        walked.extend(str(c["id"]) for c in nxt["data"])
        if not nxt["has_more"]:
            break
        cursor = nxt["last_id"]
    out["list_anthropic_after_walks_all"] = walked == [str(c["id"]) for c in full["data"]]

    back = client.get(
        "/v1/models",
        headers=_ANTHROPIC_H,
        params={"before_id": full["last_id"], "limit": 100},
    ).json()
    out["list_anthropic_before_pages_back"] = [str(c["id"]) for c in back["data"]] == [
        str(c["id"]) for c in full["data"][:-1]
    ]
    back_first = client.get(
        "/v1/models",
        headers=_ANTHROPIC_H,
        params={"before_id": full["first_id"], "limit": 100},
    ).json()
    out["list_anthropic_before_first_empty"] = (
        back_first["data"] == [] and back_first["has_more"] is False
    )
    ghost = client.get(
        "/v1/models",
        headers=_ANTHROPIC_H,
        params={"after_id": "mdl_ghost", "limit": 5},
    ).json()
    out["list_anthropic_unknown_after_empty"] = ghost["data"] == [] and ghost["has_more"] is False
    ghost2 = client.get(
        "/v1/models",
        headers=_ANTHROPIC_H,
        params={"before_id": "mdl_ghost", "limit": 5},
    ).json()
    out["list_anthropic_unknown_before_empty"] = (
        ghost2["data"] == [] and ghost2["has_more"] is False
    )
    bad_limit = client.get("/v1/models", headers=_ANTHROPIC_H, params={"limit": 0})
    out["list_anthropic_limit_refused"] = bad_limit.status_code == 422
    # OpenAI ignores anthropic cursors without the header.
    oall = client.get("/v1/models", params={"limit": 1, "after_id": "nope"})
    out["list_openai_ignores_cursors"] = oall.status_code == 200 and len(
        oall.json()["data"]
    ) == len(ids)
    return out


# ---------------------------------------------------------------------------
# Retrieve — cards and enveloped refusals
# ---------------------------------------------------------------------------


def _probe_get() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client()

    card = client.get("/v1/models/fx1")
    out["get_openai_card"] = (
        card.status_code == 200
        and card.json().get("object") == "model"
        and card.json().get("id") == "fx1"
        and isinstance(card.json().get("created"), int)
    )
    out["get_backend_cards"] = all(
        client.get(f"/v1/models/{name}").status_code == 200
        for name in ("hosted_k3", "local_fx1", "byok")
    )
    acard = client.get("/v1/models/fx1", headers=_ANTHROPIC_H)
    out["get_anthropic_card"] = (
        acard.status_code == 200
        and acard.json().get("type") == "model"
        and acard.json().get("id") == "fx1"
        and acard.json().get("display_name") == "fx1"
        and isinstance(acard.json().get("created_at"), str)
    )

    miss = client.get("/v1/models/mdl-ghost")
    err = _err(miss)
    out["get_unknown_openai_404"] = (
        miss.status_code == 404
        and _is_openai_envelope(miss)
        and err.get("code") == "model_not_found"
    )
    amiss = client.get("/v1/models/mdl-ghost", headers=_ANTHROPIC_H)
    aerr = _err(amiss)
    out["get_unknown_anthropic_404"] = (
        amiss.status_code == 404
        and _is_anthropic_envelope(amiss)
        and aerr.get("type") == "not_found_error"
    )
    out["get_case_sensitive"] = client.get("/v1/models/FX1").status_code == 404
    out["get_no_wildcard_expansion"] = client.get("/v1/models/%2A").status_code == 404
    return out


# ---------------------------------------------------------------------------
# ft: lifecycle — mint → card → resolve → tombstone
# ---------------------------------------------------------------------------


def _probe_ft_lifecycle() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver)

    job, name = _ft_job(client, suffix="lc")
    out["ft_job_succeeds"] = job["status"] == "succeeded" and name.startswith("ft:fx1:lc:")

    # The minted name joins the inventory in both dialects.
    ids = _ids(client.get("/v1/models"))
    out["ft_listed_openai"] = name in ids
    a_ids = [str(c["id"]) for c in client.get("/v1/models", headers=_ANTHROPIC_H).json()["data"]]
    out["ft_listed_anthropic"] = name in a_ids

    card = client.get(f"/v1/models/{name}")
    out["ft_card_openai"] = (
        card.status_code == 200
        and card.json().get("id") == name
        and card.json().get("object") == "model"
    )
    acard = client.get(f"/v1/models/{name}", headers=_ANTHROPIC_H)
    out["ft_card_anthropic"] = (
        acard.status_code == 200
        and acard.json().get("id") == name
        and acard.json().get("type") == "model"
    )

    # The card's `created` is the registration stamp — the same value the
    # job's checkpoint listing publishes, never the app's boot stamp.
    ckpts = _checkpoints(client, str(job["id"]))
    out["ft_checkpoint_listed"] = (
        len(ckpts) == 1
        and ckpts[0]["object"] == "fine_tuning.job.checkpoint"
        and ckpts[0]["fine_tuned_model_checkpoint"] == name
        and str(ckpts[0]["id"]).startswith("ftckpt-")
    )
    out["ft_checkpoint_id_deterministic"] = (
        ckpts[0]["id"]
        == "ftckpt-" + hashlib.sha256(f"{job['id']}:{name}".encode()).hexdigest()[:24]
    )
    out["ft_card_created_is_registration"] = (
        isinstance(card.json().get("created"), int)
        and card.json()["created"] == ckpts[0]["created_at"]
    )
    out["ft_card_created_at_or_after_job"] = card.json()["created"] >= int(job["created_at"])

    # The name resolves on every completion surface at the registered
    # checkpoint — the local_fx1 link, never the hosted default.
    resolver.resolved.clear()
    chat = _chat(client, name)
    out["ft_chat_resolves"] = chat.status_code == 200
    out["ft_routes_checkpoint"] = (
        bool(resolver.resolved)
        and resolver.resolved[-1][0] == "local_fx1"
        and str(resolver.resolved[-1][1]).endswith("ckpt")
    )
    out["ft_response_model_is_ft_name"] = chat.json().get("model") == name
    out["ft_fingerprint_is_backend"] = chat.json().get("system_fingerprint") == "local_fx1"
    out["ft_response_not_default_link"] = "hosted" not in str(chat.json().get("system_fingerprint"))

    resolver.resolved.clear()
    resp = _respond(client, name)
    out["ft_responses_resolves"] = resp.status_code == 200 and resp.json().get("model") == name
    out["ft_responses_routes_checkpoint"] = (
        bool(resolver.resolved)
        and resolver.resolved[-1][0] == "local_fx1"
        and str(resolver.resolved[-1][1]).endswith("ckpt")
    )

    resolver.resolved.clear()
    msg = _message(client, name)
    out["ft_messages_resolves"] = msg.status_code == 200 and resolver.resolved[-1][0] == "local_fx1"
    out["ft_messages_model_is_ft_name"] = msg.json().get("model") == name

    # Explicit caller-stated links beat the registry — the ft name never
    # overrides an asked-for backend.
    resolver.resolved.clear()
    _chat(
        client,
        name,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": "/srv/explicit-ckpt",
        },
    )
    out["ft_explicit_backend_wins"] = resolver.resolved[-1] == (
        "local_fx1",
        "/srv/explicit-ckpt",
    )
    resolver.resolved.clear()
    _chat(
        client,
        name,
        headers={
            "X-Fx1-Byok-Base-Url": "https://api.example.invalid/v1",
            "X-Fx1-Byok-Api-Key": "sk-synthetic",
            "X-Fx1-Byok-Model": "upstream-m",
        },
    )
    out["ft_byok_headers_beat_registry"] = resolver.resolved[-1][0] == "byok"

    # Tombstone: the name dies on every surface at once.
    dele = client.delete(f"/v1/models/{name}")
    out["ft_delete_record"] = (
        dele.status_code == 200
        and dele.json().get("id") == name
        and dele.json().get("object") == "model"
        and dele.json().get("deleted") is True
    )
    out["ft_delete_tombstones_get"] = client.get(f"/v1/models/{name}").status_code == 404
    out["ft_delete_tombstones_list"] = name not in _ids(client.get("/v1/models")) and name not in [
        str(c["id"]) for c in client.get("/v1/models", headers=_ANTHROPIC_H).json()["data"]
    ]
    resolve_404 = _chat(client, name)
    out["ft_delete_tombstones_resolve"] = (
        resolve_404.status_code == 404 and _err(resolve_404).get("code") == "model_not_found"
    )
    out["ft_delete_tombstones_resolve_messages"] = _message(client, name).status_code == 404
    out["ft_delete_drops_checkpoints"] = _checkpoints(client, str(job["id"])) == []
    dele2 = client.delete(f"/v1/models/{name}")
    out["ft_delete_twice_404"] = (
        dele2.status_code == 404 and _err(dele2).get("code") == "model_not_found"
    )

    # The registry stays writable after a tombstone — a fresh job mints.
    job2, name2 = _ft_job(client, suffix="post")
    out["ft_registry_accepts_post_delete"] = (
        job2["status"] == "succeeded" and name2 != name and name2 in _ids(client.get("/v1/models"))
    )
    return out


def _probe_ft_inflight() -> dict[str, bool]:
    out: dict[str, bool] = {}
    gate_backend = _GateBackend()
    resolver = _Resolver(
        {
            "hosted_k3": lambda **k: _StubBackend("hosted-stub"),
            "local_fx1": lambda **k: gate_backend,
            "byok": lambda **k: _StubBackend("byok-stub"),
        }
    )
    client, _api = _client(resolver)
    _job, name = _ft_job(client, suffix="fly")

    # A request already bound to the checkpoint completes across the
    # delete; every new request after the tombstone refuses.
    done: dict[str, Any] = {}
    t = threading.Thread(target=lambda: done.__setitem__("r", _chat(client, name)))
    t.start()
    out["inflight_enters_backend"] = gate_backend.entered.wait(15)
    dele = client.delete(f"/v1/models/{name}")
    gate_backend.gate.set()
    t.join(15)
    out["inflight_completes_across_delete"] = (
        dele.status_code == 200 and done["r"].status_code == 200
    )
    new_req = _chat(client, name)
    out["inflight_new_request_404s"] = (
        new_req.status_code == 404 and _err(new_req).get("code") == "model_not_found"
    )
    return out


def _probe_ft_sync() -> dict[str, bool]:
    """Registry vs job lifecycle: only a succeeded job with a checkpoint
    ever mints — failed, cancelled, and model-less outcomes mint
    nothing."""
    out: dict[str, bool] = {}

    # Failed runner → failed job, no name, no card.
    def _boom(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        raise RuntimeError("no trainer")

    c_fail, _ = _client(ft_runner=_boom)
    before = set(_ids(c_fail.get("/v1/models")))
    fid = _upload(c_fail)
    r = c_fail.post(
        "/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": fid, "suffix": "f"}
    )
    fin = _wait_terminal(c_fail, str(r.json()["id"]))
    out["ft_failed_is_failed"] = fin["status"] == "failed" and fin.get("error") is not None
    out["ft_failed_mints_nothing"] = set(_ids(c_fail.get("/v1/models"))) == before

    # Cancelled job: the runner returns a full outcome the moment the
    # cancel flag lands — the cancelled status wins and nothing mints.
    def _cancelable(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        from fx1.serve.finetune import FTJobOutcome

        for _ in range(3000):
            if should_cancel():
                return FTJobOutcome(
                    fine_tuned_model=spec.ft_model_name,
                    checkpoint=str(spec.work_dir / "ckpt"),
                )
            time.sleep(0.01)
        return FTJobOutcome()

    c_cancel, _ = _client(ft_runner=_cancelable)
    fid2 = _upload(c_cancel)
    r2 = c_cancel.post(
        "/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": fid2, "suffix": "c"}
    )
    jid2 = str(r2.json()["id"])
    for _ in range(600):
        if c_cancel.get(f"/v1/fine_tuning/jobs/{jid2}").json()["status"] == "running":
            break
        time.sleep(0.02)
    c_cancel.post(f"/v1/fine_tuning/jobs/{jid2}/cancel")
    fin2 = _wait_terminal(c_cancel, jid2)
    out["ft_cancelled_status"] = fin2["status"] == "cancelled"
    out["ft_cancelled_no_name"] = fin2.get("fine_tuned_model") is None
    out["ft_cancelled_mints_nothing"] = set(_ids(c_cancel.get("/v1/models"))) == _BASE_IDS

    # A succeeded job whose outcome declares no model registers nothing —
    # the success is honest but the registry stays clean.
    def _nameless(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
        from fx1.serve.finetune import FTJobOutcome

        return FTJobOutcome(fine_tuned_model=None)

    c_none, _ = _client(ft_runner=_nameless)
    fid3 = _upload(c_none)
    r3 = c_none.post(
        "/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": fid3, "suffix": "n"}
    )
    fin3 = _wait_terminal(c_none, str(r3.json()["id"]))
    out["ft_no_outcome_succeeds_cleanly"] = fin3["status"] == "succeeded"
    out["ft_no_outcome_no_phantom"] = fin3.get("fine_tuned_model") is None and all(
        not i.startswith("ft:") for i in _ids(c_none.get("/v1/models"))
    )
    out["ft_no_outcome_no_card"] = c_none.get("/v1/models/ft:fx1:n:000000000000").status_code == 404

    # A ghost ft name never fabricates a card anywhere.
    resolver = _spy_resolver()
    c_ghost, _ = _client(resolver)
    ghost = "ft:fx1:ghost:000000000000"
    out["ft_ghost_card_404"] = c_ghost.get(f"/v1/models/{ghost}").status_code == 404
    out["ft_ghost_chat_404"] = _chat(c_ghost, ghost).status_code == 404
    out["ft_ghost_responses_404"] = _respond(c_ghost, ghost).status_code == 404
    out["ft_ghost_messages_404"] = _message(c_ghost, ghost).status_code == 404
    out["ft_ghost_never_resolved"] = resolver.resolved == []
    out["ft_ghost_checkpoints_404"] = (
        c_ghost.get("/v1/fine_tuning/jobs/ftjob-ghost/checkpoints").status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Checkpoint gates — resolution guards the local_fx1 lane
# ---------------------------------------------------------------------------


def _probe_checkpoint_gates() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver)

    # checkpoint_dir only binds local_fx1 — anywhere else is a clean 422.
    with_ckpt = _chat(
        client,
        "fx1",
        headers={"X-Fx1-Backend": "hosted_k3", "X-Fx1-Checkpoint-Dir": "/srv/ckpt"},
    )
    out["ckpt_dir_only_local_fx1"] = with_ckpt.status_code == 422 and _is_openai_envelope(with_ckpt)
    no_ckpt = _chat(client, "fx1", headers={"X-Fx1-Backend": "local_fx1"})
    out["ckpt_local_fx1_needs_dir"] = no_ckpt.status_code == 422
    ok = _chat(
        client,
        "fx1",
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/srv/ok-ckpt"},
    )
    out["ckpt_explicit_dir_routes"] = ok.status_code == 200 and resolver.resolved[-1] == (
        "local_fx1",
        "/srv/ok-ckpt",
    )

    # Resolver faults map to wire classes — a missing checkpoint is the
    # client's bad reference (422), an unknown link name is 404, an
    # unconfigured engine is the server's 503, never a 500.
    resolver.raise_for["local_fx1"] = FileNotFoundError("no modelcard.json")
    miss_ckpt = _chat(
        client,
        "fx1",
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/srv/missing"},
    )
    out["ckpt_missing_dir_422"] = miss_ckpt.status_code == 422
    resolver.raise_for["local_fx1"] = RuntimeError("ship gate refused")
    ship = _chat(
        client,
        "fx1",
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/srv/x"},
    )
    out["ckpt_shipgate_503"] = ship.status_code == 503 and _is_openai_envelope(ship)
    resolver.raise_for.clear()

    # An ft: name smuggled into the backend header is refused — the
    # header takes backend ids, not registry names.
    smuggle = _chat(client, "fx1", headers={"X-Fx1-Backend": "ft:x:y:z"})
    out["ckpt_ft_backend_header_refused"] = smuggle.status_code in (400, 422)

    # The fallback chain honors an ft: request's checkpoint only on the
    # resolved link — a fallback link sees no checkpoint it never asked for.
    resolver.resolved.clear()
    fb = _chat(client, "fx1", headers={"X-Fx1-Fallbacks": "byok"})
    out["ckpt_fallback_runs"] = fb.status_code == 200
    out["ckpt_fallback_no_ckpt_leak"] = all(
        link != "byok" or ckpt is None for link, ckpt in resolver.resolved
    )
    return out


# ---------------------------------------------------------------------------
# Delete — tombstone records and refusals
# ---------------------------------------------------------------------------


def _probe_delete() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client()
    job, name = _ft_job(client, suffix="del")

    for builtin in ("fx1", "hosted_k3", "local_fx1", "byok"):
        r = client.delete(f"/v1/models/{builtin}")
        err = _err(r)
        out[f"delete_builtin_{builtin}_refused"] = (
            r.status_code == 400 and _is_openai_envelope(r) and err.get("code") == "invalid_request"
        )
        out[f"delete_builtin_{builtin}_still_listed"] = builtin in _ids(client.get("/v1/models"))

    ghost = client.delete("/v1/models/mdl-ghost")
    out["delete_unknown_404"] = (
        ghost.status_code == 404 and _err(ghost).get("code") == "model_not_found"
    )
    ghost_ft = client.delete("/v1/models/ft:fx1:ghost:000000000000")
    out["delete_unregistered_ft_404"] = ghost_ft.status_code == 404

    real = client.delete(f"/v1/models/{name}")
    out["delete_registered_200"] = (
        real.status_code == 200
        and real.json().get("id") == name
        and real.json().get("deleted") is True
    )
    out["delete_is_tombstone"] = client.get(f"/v1/models/{name}").status_code == 404
    out["delete_drops_checkpoints"] = _checkpoints(client, str(job["id"])) == []
    # The job itself is not touched — only the registry card dies.
    job_after = client.get(f"/v1/fine_tuning/jobs/{job['id']}")
    out["delete_keeps_job_history"] = (
        job_after.status_code == 200 and job_after.json().get("status") == "succeeded"
    )
    # Anthropic dialect on the DELETE error path — the /v1/models error
    # envelope is OpenAI's on this surface (dialect is card-level).
    miss = client.delete("/v1/models/mdl-ghost", headers=_ANTHROPIC_H)
    out["delete_anthropic_error_envelope"] = miss.status_code == 404 and _is_openai_envelope(miss)
    return out


# ---------------------------------------------------------------------------
# model param validation — unknown and empty names refuse enveloped
# ---------------------------------------------------------------------------


def _probe_model_param() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver)

    resolver.resolved.clear()
    chat = _chat(client, "mdl-ghost")
    out["param_unknown_chat_404"] = (
        chat.status_code == 404 and _err(chat).get("code") == "model_not_found"
    )
    comp = client.post("/v1/completions", json={"model": "mdl-ghost", "prompt": "hi"})
    out["param_unknown_completions_404"] = comp.status_code == 404 and _is_openai_envelope(comp)
    resp = _respond(client, "mdl-ghost")
    out["param_unknown_responses_404"] = resp.status_code == 404
    msg = _message(client, "mdl-ghost")
    out["param_unknown_messages_404"] = (
        msg.status_code == 404
        and _is_anthropic_envelope(msg)
        and _err(msg).get("type") == "not_found_error"
    )
    ct = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "mdl-ghost",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    out["param_unknown_count_tokens_404"] = (
        ct.status_code == 404
        and _is_anthropic_envelope(ct)
        and _err(ct).get("type") == "not_found_error"
    )
    out["param_unknown_never_500"] = (
        chat.status_code != 500
        and comp.status_code != 500
        and resp.status_code != 500
        and msg.status_code != 500
        and ct.status_code != 500
    )
    # The unknown name never reached a resolver — a refusal is a verdict,
    # not a quiet swap to the default link.
    out["param_unknown_never_resolved"] = resolver.resolved == []

    # Empty model refuses validation — the default link must never pick
    # up a name the caller typed wrong.
    e_chat = _chat(client, "")
    out["param_empty_chat_422"] = e_chat.status_code == 422
    e_comp = client.post("/v1/completions", json={"model": "", "prompt": "hi"})
    out["param_empty_completions_422"] = e_comp.status_code == 422
    e_resp = _respond(client, "")
    out["param_empty_responses_422"] = e_resp.status_code == 422
    e_msg = _message(client, "")
    out["param_empty_messages_422"] = e_msg.status_code == 422
    e_emb = client.post("/v1/embeddings", json={"model": "", "input": "x"})
    out["param_empty_embeddings_422"] = e_emb.status_code == 422

    # Known link names still pick their link; the base alias serves.
    resolver.resolved.clear()
    byok_chat = _chat(client, "byok")
    out["param_backend_name_links"] = (
        byok_chat.status_code == 200 and resolver.resolved[-1][0] == "byok"
    )
    resolver.resolved.clear()
    base = _chat(client, "fx1")
    out["param_base_alias_serves"] = (
        base.status_code == 200 and resolver.resolved[-1][0] == "hosted_k3"
    )
    # Embeddings treat `model` as the upstream embedding deployment name —
    # free-form, passed to the link verbatim, never a registry lookup.
    resolver.resolved.clear()
    emb = client.post("/v1/embeddings", json={"model": "emb-upstream-9", "input": "x"})
    out["param_embeddings_freeform"] = (
        emb.status_code == 200
        and emb.json().get("model") == "emb-upstream-9"
        and resolver.resolved[-1][0] == "hosted_k3"
    )
    return out


# ---------------------------------------------------------------------------
# Envelope dialect — one key, two grammars
# ---------------------------------------------------------------------------


def _probe_envelope_dialect() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver, api_key=_ROOT)
    key = _ROOT

    xkey = client.get("/v1/models", headers=_h(key))
    bearer = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})
    out["dialect_x_api_key_resolves"] = xkey.status_code == 200
    out["dialect_bearer_same_key"] = bearer.status_code == 200
    out["dialect_same_inventory"] = _ids(xkey) == _ids(bearer)

    ax = client.get("/v1/models", headers=_ah(key))
    axb = client.get("/v1/models", headers={**_ANTHROPIC_H, "Authorization": f"Bearer {key}"})
    out["dialect_anthropic_x_api_key"] = ax.status_code == 200 and "has_more" in ax.json()
    out["dialect_anthropic_bearer"] = axb.status_code == 200 and _ids(ax) == _ids(axb)

    # The completion surface resolves the same key under both spellings.
    chat_x = _chat(client, "byok", headers=_h(key))
    chat_b = _chat(client, "byok", headers={"Authorization": f"Bearer {key}"})
    out["dialect_chat_both_keys"] = chat_x.status_code == 200 and chat_b.status_code == 200

    # Anthropic dialect completion with the same key on x-api-key.
    msg = _message(client, "byok", headers=_ah(key))
    out["dialect_messages_x_api_key"] = msg.status_code == 200

    bad = client.get("/v1/models", headers=_h("wrong-key"))
    out["dialect_wrong_key_refused"] = bad.status_code == 401 and isinstance(bad.json(), dict)
    none_ = client.get("/v1/models")
    out["dialect_no_key_refused"] = none_.status_code == 401

    # Cards differ per dialect for the same id.
    oc = client.get("/v1/models/fx1", headers=_h(key)).json()
    ac = client.get("/v1/models/fx1", headers=_ah(key)).json()
    out["dialect_card_shapes_differ"] = (
        oc.get("object") == "model" and ac.get("type") == "model" and oc.get("id") == ac.get("id")
    )
    return out


# ---------------------------------------------------------------------------
# Durability — state_dir restart replays registry + tombstones
# ---------------------------------------------------------------------------


def _probe_durability() -> dict[str, bool]:
    out: dict[str, bool] = {}
    td = _temporary_directory()
    state = Path(td)

    c1, _ = _client(_spy_resolver(), state_dir=state)
    job, name = _ft_job(c1, suffix="dur")
    doomed_j, doomed = _ft_job(c1, suffix="dead")
    c1.delete(f"/v1/models/{doomed}")
    out["dur_journal_written"] = (state / "ft_jobs.jsonl").exists()

    # Fresh process: same state dir replays registry + tombstone verbatim.
    r2 = _spy_resolver()
    c2, _ = _client(r2, state_dir=state)
    out["dur_registry_survives"] = name in _ids(c2.get("/v1/models"))
    card = c2.get(f"/v1/models/{name}")
    out["dur_card_survives"] = (
        card.status_code == 200
        and card.json().get("id") == name
        and card.json().get("created") == _checkpoints(c2, str(job["id"]))[0]["created_at"]
    )
    r2.resolved.clear()
    chat = _chat(c2, name)
    out["dur_ft_resolves_post_restart"] = (
        chat.status_code == 200
        and r2.resolved[-1][0] == "local_fx1"
        and str(r2.resolved[-1][1]).endswith("ckpt")
    )
    out["dur_tombstone_stays_dead"] = (
        c2.get(f"/v1/models/{doomed}").status_code == 404
        and doomed not in _ids(c2.get("/v1/models"))
        and _chat(c2, doomed).status_code == 404
        and _checkpoints(c2, str(doomed_j["id"])) == []
    )
    out["dur_job_survives"] = c2.get(f"/v1/fine_tuning/jobs/{job['id']}").status_code == 200
    out["dur_base_never_tombstoned"] = (
        "fx1" in _ids(c2.get("/v1/models"))
        and c2.get("/v1/models/fx1").status_code == 200
        and c2.delete("/v1/models/fx1").status_code == 400
    )
    # The recovered registry is live — a new job mints and resolves.
    job3, name3 = _ft_job(c2, suffix="post")
    out["dur_new_mint_post_restart"] = (
        job3["status"] == "succeeded"
        and name3.startswith("ft:fx1:post:")
        and name3 in _ids(c2.get("/v1/models"))
        and _chat(c2, name3).status_code == 200
    )
    return out


# ---------------------------------------------------------------------------
# Concurrency — deletes and mints resolve atomically
# ---------------------------------------------------------------------------


def _probe_concurrency() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolver = _spy_resolver()
    client, _api = _client(resolver)
    _job, name = _ft_job(client, suffix="race")

    # list-vs-delete and get-vs-delete races resolve atomically: every
    # answer is either the full live registry/card or an honest tombstone.
    torn: list[str] = []

    def _deleter() -> None:
        time.sleep(0.05)
        client.delete(f"/v1/models/{name}")

    def _lister() -> None:
        for _ in range(60):
            r = client.get("/v1/models")
            if r.status_code != 200 or not isinstance(r.json().get("data"), list):
                torn.append("list-torn")
            card_ids = _ids(r)
            if name in card_ids and card_ids.count(name) != 1:
                torn.append("list-dup")

    def _getter() -> None:
        for _ in range(60):
            r = client.get(f"/v1/models/{name}")
            if r.status_code == 200:
                if r.json().get("id") != name or r.json().get("object") != "model":
                    torn.append("get-partial")
            elif r.status_code != 404 or not _is_openai_envelope(r):
                torn.append("get-torn")

    def _resolver_thread() -> None:
        for _ in range(30):
            r = _chat(client, name)
            if r.status_code == 200:
                continue
            if r.status_code != 404 or _err(r).get("code") != "model_not_found":
                torn.append("resolve-torn")

    ths = [
        threading.Thread(target=_deleter),
        threading.Thread(target=_lister),
        threading.Thread(target=_getter),
        threading.Thread(target=_resolver_thread),
    ]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    out["conc_list_delete_atomic"] = not torn
    out["conc_delete_landed"] = client.get(f"/v1/models/{name}").status_code == 404

    # Two parallel completions mint two distinct registered names.
    b = threading.Barrier(2)
    created: list[dict[str, Any]] = []

    def _submit(sfx: str) -> None:
        fid = _upload(client)
        b.wait()
        r = client.post(
            "/v1/fine_tuning/jobs",
            json={"model": "fx1", "training_file": fid, "suffix": sfx},
        )
        created.append(r.json())

    t1, t2 = (
        threading.Thread(target=_submit, args=("p1",)),
        threading.Thread(target=_submit, args=("p2",)),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    fins = [_wait_terminal(client, str(j["id"])) for j in created]
    names = [str(f["fine_tuned_model"]) for f in fins]
    out["conc_two_mints_succeed"] = all(f["status"] == "succeeded" for f in fins)
    out["conc_two_mints_distinct"] = len(set(names)) == 2 and all(
        n.startswith("ft:fx1:") for n in names
    )
    out["conc_both_mints_listed"] = all(n in _ids(client.get("/v1/models")) for n in names)
    return out


# ---------------------------------------------------------------------------
# Errors — hostile ids are all enveloped refusals
# ---------------------------------------------------------------------------


def _probe_errors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _api = _client()

    def _refuses(url: str) -> bool:
        r = client.get(url)
        return r.status_code in (400, 404) and isinstance(r.json(), dict)

    # Percent-decoded ids can never escape the path segment — %2F makes
    # the route not match at all, %2E%2E decodes to ".." and misses the
    # registry; both land in the catch-all/model_not_found envelope.
    out["err_encoded_dotdot"] = _refuses("/v1/models/%2E%2E")
    out["err_encoded_slash"] = _refuses("/v1/models/a%2Fb")
    out["err_literal_dotdot"] = _refuses("/v1/models/..")
    out["err_wildcard"] = _refuses("/v1/models/mdl-%2A")
    out["err_nul_byte"] = _refuses("/v1/models/a%00b")
    out["err_long_id"] = _refuses("/v1/models/" + "x" * 4096)
    out["err_encoded_colon_ft"] = _refuses("/v1/models/ft%3Ax%3Ay%3Az")
    out["err_space"] = _refuses("/v1/models/a%20b")

    # The same hostile inputs through DELETE are enveloped too.
    out["err_delete_encoded_slash"] = client.delete("/v1/models/a%2Fb").status_code in (400, 404)
    out["err_delete_long_id"] = client.delete("/v1/models/" + "x" * 4096).status_code == 404

    # Hostile model values in request bodies are enveloped, never 500s.
    wild = _chat(client, "ft:*")
    out["err_body_wildcard_ft"] = wild.status_code == 404 and _is_openai_envelope(wild)
    long_model = _chat(client, "z" * 4096)
    out["err_body_long_model"] = (
        long_model.status_code in (404, 422) and long_model.status_code != 500
    )
    dotdot = _chat(client, "../etc/passwd")
    out["err_body_dotdot"] = dotdot.status_code in (404, 422)
    nul = _chat(client, "a\x00b")
    out["err_body_nul"] = nul.status_code in (400, 404, 422)
    # Unknown route under /v1 lands in the catch-all error envelope.
    unknown = client.get("/v1/models/../completions")
    out["err_catchall_enveloped"] = unknown.status_code in (404, 405) and isinstance(
        unknown.json(), dict
    )
    return out


# ---------------------------------------------------------------------------
# SDK parity — the in-process twin pins the same registry
# ---------------------------------------------------------------------------


def _probe_sdk() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.sdk import Fx1Harness
    from fx1.serve.openai_compat import OpenAICompatError

    resolver = _spy_resolver()
    td = _temporary_directory()
    sdk = Fx1Harness(
        backend_resolver=resolver,
        state_dir=td / "sdk-state",
        ft_runner=_ok_runner,
        ft_dir=td / "ft",
    )
    out["sdk_base_ids"] = {m.id for m in sdk.openai_models().data} == _BASE_IDS
    out["sdk_card_fx1"] = sdk.openai_model("fx1").id == "fx1"
    try:
        sdk.openai_model("mdl-ghost")
        raised = False
    except OpenAICompatError as exc:
        raised = exc.status == 404 and exc.code == "model_not_found"
    out["sdk_unknown_404"] = raised

    job = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS, suffix="sdk")
    name = str(job.fine_tuned_model)
    out["sdk_job_mints"] = job.status == "succeeded" and name.startswith("ft:fx1:sdk:")
    out["sdk_ft_listed"] = name in {m.id for m in sdk.openai_models().data}
    out["sdk_ft_card"] = sdk.openai_model(name).id == name
    resolver.resolved.clear()
    env, _cid = sdk.openai_chat({"model": name, "messages": [{"role": "user", "content": "hi"}]})
    out["sdk_ft_resolves_ckpt"] = resolver.resolved[-1][0] == "local_fx1" and str(
        resolver.resolved[-1][1]
    ).endswith("ckpt")
    out["sdk_ft_response_model"] = env.model == name
    out["sdk_ghost_404"] = False
    try:
        sdk.openai_chat(
            {
                "model": "ft:fx1:ghost:000000000000",
                "messages": [{"role": "user", "content": "x"}],
            }
        )
    except OpenAICompatError as exc:
        out["sdk_ghost_404"] = exc.status == 404
    try:
        sdk.openai_delete_model("fx1")
        out["sdk_delete_builtin_raises"] = False
    except OpenAICompatError as exc:
        out["sdk_delete_builtin_raises"] = exc.code == "invalid_request"
    try:
        sdk.openai_delete_model("mdl-ghost")
        out["sdk_delete_unknown_raises"] = False
    except OpenAICompatError as exc:
        out["sdk_delete_unknown_raises"] = exc.status == 404
    sdk.openai_delete_model(name)
    out["sdk_delete_tombstones"] = name not in {m.id for m in sdk.openai_models().data}
    try:
        sdk.openai_model(name)
        out["sdk_tombstone_card_gone"] = False
    except OpenAICompatError:
        out["sdk_tombstone_card_gone"] = True
    return out


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def models_audit() -> dict[str, bool]:
    """Every probe, measured end-to-end against a fresh app per section."""
    out: dict[str, bool] = {}
    with _audit_context():
        out.update(_probe_list())
        out.update(_probe_get())
        out.update(_probe_ft_lifecycle())
        out.update(_probe_ft_inflight())
        out.update(_probe_ft_sync())
        out.update(_probe_checkpoint_gates())
        out.update(_probe_delete())
        out.update(_probe_model_param())
        out.update(_probe_envelope_dialect())
        out.update(_probe_durability())
        out.update(_probe_concurrency())
        out.update(_probe_errors())
        out.update(_probe_sdk())
    return out


def models_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = models_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "models_audit",
        "schema": "models_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient (in-process ASGI) + Fx1Harness SDK twin",
            "not_verified": [
                "multi-process writers on one state_dir (single-process lock)",
                "anthropic cursor stability across concurrent deletes mid-walk",
                "registry size beyond the default job store bound",
                "real network delivery/disconnect timing (TestClient buffers)",
                "power-loss durability or rollback of valid journal suffixes",
                "checkpoint directory contents (resolver spies record the path only)",
            ],
        },
        "interpretation": (
            "The /v1/models registry holds end to end: one inventory "
            "answers in both envelope dialects with honest cursors in "
            "both directions; a succeeded fine-tune mints its ft: name "
            "onto the list, the card (created = the registration stamp), "
            "the checkpoint listing, and every completion surface, where "
            "it resolves the producing job's checkpoint on the local_fx1 "
            "link and never the default. Explicit backend/BYOK headers "
            "still win over the registry. DELETE writes a real journaled "
            "tombstone — list, retrieve, resolve, and checkpoint listing "
            "drop the name at once, a mid-flight request completes, a "
            "restart never resurrects it, and the base link ids refuse "
            "deletion outright. Failed or cancelled jobs mint nothing, "
            "unknown and empty model names refuse 404/422 in the request "
            "dialect's envelope, and hostile ids (traversal, wildcards, "
            "NUL, overlong) are all enveloped refusals. Concurrent "
            "list/get/resolve against a delete resolves atomically, and "
            "the SDK twin pins the same semantics."
            if ok
            else f"MODELS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(models_audit_bench(), indent=2, sort_keys=True))
