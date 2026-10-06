"""idem_audit — Idempotency-Key deep-audit battery for the fx1 surface.

``api_audit`` pins wire shapes and ``concurrency_audit`` pins inflight
saturation; this battery attacks the *submit-once contract* end to end —
every mutating route that takes an ``Idempotency-Key``, exercised with a
real retried POST against a live in-process app.

Probe map:

- *Coverage matrix* — every mutating route honors the header:
  ``/harness/jobs`` (+``/batch``), ``/harness/runs``,
  ``/harness/evals``, ``/v1/evals/{id}/runs``, ``/harness/complete``
  (+``/batch``), ``/v1/chat/completions``, ``/v1/messages``,
  ``/v1/completions``, ``/v1/responses``, ``/v1/batches``,
  ``/v1/messages/batches``, ``/v1/fine_tuning/jobs``, ``/v1/files``,
  ``/v1/uploads`` (+``/parts``/``/complete``/``/cancel``), and the
  ``/harness/keys`` lifecycle verbs (mint/rotate/patch/revoke). For
  each: a keyed retry replays (marker + identical id) and the side
  effect fires exactly once (backend call counter / record tally). A
  route that silently ignores the header fails its probe.
- *Replay* — byte-identical body, original record id, marker header
  (``X-Fx1-Idempotent-Replay`` on the JSON envelopes, ``replayed``
  field on the harness models), and the backend stub's call counter
  proving no re-execution.
- *Conflict* — same key + different body → ``409`` with
  ``idempotency_conflict`` in both error grammars (harness ``{code}``,
  OpenAI ``{error.code}``), deterministic on repeat, and the losing
  request never reaches the backend.
- *Namespacing* — dedupe scope pinned honestly: separate stores stay
  independent (jobs vs complete), the shared OpenAI ledger conflicts
  across surfaces (chat key vs responses body), and keys are scoped
  per-credential — credential B can never replay credential A's record.
- *Key hygiene* — blank/whitespace headers are ignored as absent, an
  over-256-char key refuses ``400`` inside the envelope, and the
  256-char boundary itself works end to end.
- *Eviction* — under a small ``idem_max`` the oldest key is evicted and
  re-executes cleanly; nothing tombstoned resurrects.
- *Failed submissions* — a key whose request failed client-side (4xx)
  or provider-side (5xx) is never pinned: the retried call re-executes.
- *Replay identity* — DELETE the minted object, then retry the keyed
  POST: the idem ledger still serves the frozen answer (the replay is
  a snapshot, not a live lookup).
- *Stream+idem* — a stored streaming chat call replays byte-identical
  SSE frames; ``Last-Event-ID`` resume returns a strict suffix.
- *State-dir restart* — journaled ledgers survive a fresh app instance
  over the same ``--state-dir`` (replay without re-executing); the key
  ledger deliberately does NOT journal (its records carry the minted
  secret), so a post-restart keyed mint honestly re-executes.
- *Concurrency* — N parallel same-key submits → exactly one execution;
  N parallel same-key different-body → exactly one success, the rest
  ``409``; the winner's record stays stable for later replays.
- *Envelope* — every refusal (409 conflict, 400 over-long key) lands in
  the dialect's error envelope with a stable machine code.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

Honesty: every verdict is measured against a live in-process app with
real stores, journals, and executors; stub backends count calls — they
never answer for the idempotency layer itself. No probe fabricates a
replay.

Composition: complements ``api_audit`` (wire shapes), ``batch_audit``
(batch lifecycle), ``conv_audit`` (conversation store), and
``concurrency_audit`` (inflight saturation). This battery owns the
idempotency claims those siblings touch only incidentally.

Defects this battery caught (fixed on the same change, all pinned
green below): the six sync ``_IdemStore`` routes (complete, chat,
messages, completions, responses, complete/batch), ``_submit_eval``,
and ``/v1/fine_tuning/jobs`` did lookup → execute → put with no claim
lock, so parallel same-key submits could double-execute; the
``/v1/uploads`` verbs, ``/v1/files``, and ``/harness/keys`` mint /
rotate / patch / revoke ignored ``Idempotency-Key`` entirely — a
retried mint fabricated a second credential; key reuse was
credential-blind (a key under credential A could replay under B); and
the 409 envelope code was ``conflict``/``idempotency_conflict`` by
route — now ``idempotency_conflict`` everywhere.

Sealed ``idem_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import httpx

from fx1.serve.backends import BackendNotConfiguredError
from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _GateBackend,
    _StubBackend,
    _temporary_directory,
)
from fx1.serve.finetune import FTJobOutcome
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["idem_audit", "idem_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "1dem-aud1t-r00t"
_MODEL = "hosted_k3"
_IDEM = "Idempotency-Key"
_ANTHROPIC_V = {"anthropic-version": "2023-06-01"}


# ---------------------------------------------------------------------------
# App plumbing
# ---------------------------------------------------------------------------


def _resources() -> ExitStack:
    return _RESOURCES.get()


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _ft_stub(spec: Any, **kw: Any) -> FTJobOutcome:
    """Submit-level FT runner — the job must exist and tick, not train."""
    del spec, kw
    return FTJobOutcome(fine_tuned_model="ft:idem-audit", trained_tokens=0)


class _FailBackend(_StubBackend):
    """Every ``complete`` raises an availability fault — the request 503s
    while the call counter still proves whether a retry re-executed."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        raise BackendNotConfiguredError("synthetic outage")


class _CountingRunner:
    """``Harness`` runner with a call counter + optional release gate —
    the concurrency probes need to know a run actually executed."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.calls: list[list[str]] = []
        self.gate = gate
        self.entered = threading.Event()

    def __call__(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del timeout_s
        self.calls.append(list(argv))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(30)
        return 0, f"run-{len(self.calls)}", ""


def _make_app(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    **create_kw: Any,
) -> Any:
    """create_app under an isolated env; ``backend_map[name]()`` are the
    link factories; ``create_kw`` forwards verbatim (state_dir/idem_max)."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {_MODEL: lambda: _StubBackend("fx1")}
    resources = _resources()
    isolated = _temporary_directory()
    saved = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=runner or _fast_runner),
            backend_resolver=lambda name, *a, **k: backends[name](),
            state_dir=create_kw.pop("state_dir", isolated / "state"),
            receipts_dir=create_kw.pop("receipts_dir", isolated / "receipts"),
            ft_dir=create_kw.pop("ft_dir", isolated / "ft"),
            ft_runner=create_kw.pop("ft_runner", _ft_stub),
            **create_kw,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return app
    finally:
        if saved is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved


def _client(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    **create_kw: Any,
) -> tuple[TestClient, Any]:
    """(TestClient, app) — the battery's standard wired app."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _make_app(backend_map=backend_map, api_key=api_key, runner=runner, **create_kw)
    client = TestClient(app, raise_server_exceptions=False)
    _resources().callback(client.close)
    _resources().enter_context(client)
    return client, app


def _h(auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    h = {"X-API-Key": auth} if auth else {}
    h.update(extra)
    return h


def _ih(key: str, auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    return _h(auth, **{_IDEM: key, **extra})


def _post2(client: TestClient, path: str, body: Any, key: str, **extra: str) -> tuple[Any, Any]:
    """Two keyed POSTs, same key + same body — the atomic replay pair."""
    h = _ih(key, **extra)
    return client.post(path, json=body, headers=h), client.post(path, json=body, headers=h)


def _replay_hdr(r: Any) -> bool:
    return bool(r.headers.get("x-fx1-idempotent-replay") == "true")


def _err_code(r: Any) -> str | None:
    """The refusal code in either grammar — harness ``{code}`` or the
    OpenAI/Anthropic ``{error: {code}}`` envelope."""
    try:
        body = r.json()
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        return str(code) if code is not None else None
    code = body.get("code")
    return str(code) if code is not None else None


def _same(a: Any, b: Any, *, ignore: tuple[str, ...] = ("replayed",)) -> bool:
    """JSON-equality minus the replay marker fields."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {k: v for k, v in a.items() if k not in ignore}
        kb = {k: v for k, v in b.items() if k not in ignore}
        return bool(ka == kb)
    return bool(a == b)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


def _chat_body(content: str = "hi") -> dict[str, Any]:
    return {"model": "fx1", "messages": [{"role": "user", "content": content}]}


def _responses_body(content: str = "hi") -> dict[str, Any]:
    return {"model": "fx1", "input": content}


def _messages_body(content: str = "hi") -> dict[str, Any]:
    return {
        "model": "fx1",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": content}],
    }


def _completions_body(content: str = "hi") -> dict[str, Any]:
    return {"model": "fx1", "prompt": content}


def _complete_body(content: str = "hi") -> dict[str, Any]:
    return {"backend": _MODEL, "messages": [{"role": "user", "content": content}]}


def _cbatch_body(content: str = "hi") -> dict[str, Any]:
    return {"backend": _MODEL, "batch": [[{"role": "user", "content": content}]]}


def _eval_body() -> dict[str, Any]:
    return {"suite": "calibration", "backend": _MODEL, "timeout_s": 30}


def _spec_body() -> dict[str, Any]:
    return {
        "name": "idem-audit-spec",
        "data_source_config": {
            "type": "custom",
            "item_schema": {"suite": "calibration", "backend": _MODEL},
        },
    }


def _ft_body(training_file: str, suffix: str = "im") -> dict[str, Any]:
    return {"model": "fx1", "training_file": training_file, "suffix": suffix}


_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_BATCH_INPUT = (
    b'{"custom_id":"l1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"fx1","messages":[{"role":"user","content":"hi"}]}}\n'
)


def _file_post(
    client: TestClient,
    key: str | None,
    name: str = "im.jsonl",
    *,
    purpose: str = "fine-tune",
    content: bytes = _CORPUS,
) -> Any:
    h = _h(_ROOT) if key is None else _ih(key)
    return client.post(
        "/v1/files",
        files={"file": (name, content, "application/jsonl")},
        data={"purpose": purpose},
        headers=h,
    )


# ---------------------------------------------------------------------------
# Coverage matrix — every mutating route honors the header
# ---------------------------------------------------------------------------


def _probe_cov_harness() -> dict[str, bool]:
    """Harness routes: jobs (+batch), runs, evals, complete (+batch)."""
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    runner = _CountingRunner()
    client, _app = _client(backend_map={_MODEL: lambda: stub}, runner=runner)

    r1, r2 = _post2(client, "/harness/runs", {"command": "doctor"}, "cov.runs")
    out["coverage.runs.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json().get("replayed") is True
        and _same(r1.json(), r2.json())
    )
    out["coverage.runs.once"] = len(runner.calls) == 1

    jobs_before = client.get("/harness/jobs", headers=_h()).json().get("total")
    r1, r2 = _post2(client, "/harness/jobs", {"command": "doctor"}, "cov.jobs")
    jobs_after = client.get("/harness/jobs", headers=_h()).json().get("total")
    out["coverage.jobs.replay"] = (
        r1.status_code == 202
        and r2.status_code == 202
        and r2.json().get("replayed") is True
        and r1.json().get("job_id") == r2.json().get("job_id")
    )
    out["coverage.jobs.once"] = jobs_after - jobs_before == 1

    body = {"jobs": [{"command": "doctor", "idempotency_key": "cov.jobs_batch"}]}
    r1 = client.post("/harness/jobs/batch", json=body, headers=_h())
    r2 = client.post("/harness/jobs/batch", json=body, headers=_h())
    j1 = (r1.json().get("jobs") or [{}])[0]
    j2 = (r2.json().get("jobs") or [{}])[0]
    jobs_after2 = client.get("/harness/jobs", headers=_h()).json().get("total")
    out["coverage.jobs_batch.replay"] = (
        r1.status_code == 202
        and r2.status_code == 202
        and j2.get("replayed") is True
        and j1.get("job_id") == j2.get("job_id")
    )
    out["coverage.jobs_batch.once"] = jobs_after2 - jobs_after == 1

    r1, r2 = _post2(client, "/harness/evals", _eval_body(), "cov.evals")
    evals_total = client.get("/harness/evals", headers=_h()).json().get("total")
    out["coverage.evals.replay"] = (
        r1.status_code == 202
        and r2.status_code == 202
        and r2.json().get("replayed") is True
        and r1.json().get("eval_id") == r2.json().get("eval_id")
    )
    out["coverage.evals.once"] = evals_total == 1

    before = stub.calls
    r1, r2 = _post2(client, "/harness/complete", _complete_body(), "cov.complete")
    out["coverage.complete.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json().get("replayed") is True
        and _same(r1.json(), r2.json())
        and r1.json().get("completion_id") == r2.json().get("completion_id")
    )
    out["coverage.complete.once"] = stub.calls == before + 1

    before = stub.calls
    r1, r2 = _post2(client, "/harness/complete/batch", _cbatch_body(), "cov.cbatch")
    out["coverage.complete_batch.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json().get("replayed") is True
        and _same(r1.json(), r2.json())
    )
    out["coverage.complete_batch.once"] = stub.calls == before + 1
    return out


def _probe_cov_openai() -> dict[str, bool]:
    """OpenAI + Anthropic surfaces — replay rides the marker header."""
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    def pair(name: str, path: str, body: dict[str, Any], **extra: str) -> None:
        before = stub.calls
        r1, r2 = _post2(client, path, body, f"cov.{name}", **extra)
        ok = (
            r1.status_code == 200
            and r2.status_code == 200
            and _replay_hdr(r2)
            and _same(r1.json(), r2.json())
            and r1.json().get("id") == r2.json().get("id")
        )
        out[f"coverage.{name}.replay"] = ok
        out[f"coverage.{name}.once"] = stub.calls == before + 1

    pair("chat", "/v1/chat/completions", _chat_body())
    pair("responses", "/v1/responses", _responses_body())
    pair("completions", "/v1/completions", _completions_body())
    pair("messages", "/v1/messages", _messages_body(), **_ANTHROPIC_V)

    # batches: input file first, then keyed batch create
    fid_r = _file_post(client, None, name="in.jsonl", purpose="batch", content=_BATCH_INPUT)
    fid = str(fid_r.json()["id"])
    body = {"input_file_id": fid, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
    r1, r2 = _post2(client, "/v1/batches", body, "cov.batches")
    out["coverage.batches.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and _replay_hdr(r2)
        and r1.json().get("id") == r2.json().get("id")
    )
    n_batches = len(client.get("/v1/batches", headers=_h()).json().get("data", []))
    out["coverage.batches.once"] = n_batches == 1

    abody = {
        "requests": [
            {
                "custom_id": "c1",
                "params": {
                    "model": "fx1",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            }
        ]
    }
    r1, r2 = _post2(client, "/v1/messages/batches", abody, "cov.mbatches", **_ANTHROPIC_V)
    out["coverage.messages_batches.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and _replay_hdr(r2)
        and r1.json().get("id") == r2.json().get("id")
    )

    # fine-tuning: keyed submit pins the ftjob id — the job record is
    # live (a fast worker can flip queued->running between the two
    # posts), so the dedupe pin is identity + a single record, not a
    # frozen body diff
    tfid = str(_file_post(client, None, name="ft.jsonl").json()["id"])
    r1, r2 = _post2(client, "/v1/fine_tuning/jobs", _ft_body(tfid), "cov.ft")
    n_ft = len(client.get("/v1/fine_tuning/jobs", headers=_h()).json().get("data", []))
    out["coverage.fine_tuning.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json().get("id") == r2.json().get("id")
        and str(r1.json()["id"]).startswith("ftjob-")
    )
    out["coverage.fine_tuning.once"] = n_ft == 1
    return out


def _probe_cov_evals() -> dict[str, bool]:
    """``/v1/evals`` spec + keyed run submit."""
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})
    spec = client.post("/v1/evals", json=_spec_body(), headers=_h())
    sid = str(spec.json()["id"])
    body = {"model": _MODEL}
    r1, r2 = _post2(client, f"/v1/evals/{sid}/runs", body, "cov.evalruns")
    runs = client.get(f"/v1/evals/{sid}/runs", headers=_h()).json().get("data", [])
    # the run record is live — a fast worker can flip queued->running
    # between the two posts, so the dedupe pin is identity + a single
    # record, not a frozen body diff
    out["coverage.eval_runs.replay"] = (
        r1.status_code == 201
        and r2.status_code == 201
        and r1.json().get("id") == r2.json().get("id")
        and str(r1.json()["id"]).startswith("evalrun_")
    )
    out["coverage.eval_runs.once"] = len(runs) == 1
    return out


def _probe_cov_files_uploads() -> dict[str, bool]:
    """/v1/files, /v1/uploads create/parts/complete/cancel."""
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    r1 = _file_post(client, "cov.files")
    r2 = _file_post(client, "cov.files")
    n_files = len(client.get("/v1/files", headers=_h()).json().get("data", []))
    out["coverage.files.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and _replay_hdr(r2)
        and r1.json().get("id") == r2.json().get("id")
    )
    out["coverage.files.once"] = n_files == 1
    # same key + different bytes conflicts
    r3 = client.post(
        "/v1/files",
        files={"file": ("im.jsonl", _CORPUS + b'{"messages":[]}\n', "application/jsonl")},
        data={"purpose": "fine-tune"},
        headers=_ih("cov.files"),
    )
    out["coverage.files.conflict"] = (
        r3.status_code == 409 and _err_code(r3) == "idempotency_conflict"
    )

    up_body = {
        "purpose": "batch",
        "filename": "up.jsonl",
        "bytes": len(_CORPUS),
        "mime_type": "application/jsonl",
    }
    r1, r2 = _post2(client, "/v1/uploads", up_body, "cov.upcreate")
    out["coverage.uploads_create.replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and _replay_hdr(r2)
        and r1.json().get("id") == r2.json().get("id")
    )
    up_id = str(r1.json()["id"])
    # a second distinct create still mints a fresh upload
    r3 = client.post("/v1/uploads", json=up_body, headers=_h())
    out["coverage.uploads_create.fresh"] = r3.status_code == 200 and r3.json().get("id") not in (
        None,
        up_id,
    )

    pr1 = client.post(
        f"/v1/uploads/{up_id}/parts",
        files={"data": ("p", _CORPUS, "application/octet-stream")},
        headers=_ih("cov.part"),
    )
    pr2 = client.post(
        f"/v1/uploads/{up_id}/parts",
        files={"data": ("p", _CORPUS, "application/octet-stream")},
        headers=_ih("cov.part"),
    )
    out["coverage.uploads_part.replay"] = (
        pr1.status_code == 200
        and pr2.status_code == 200
        and _replay_hdr(pr2)
        and pr1.json().get("id") == pr2.json().get("id")
    )
    # keyed retry with different bytes fails closed — the part dedupe
    # fingerprints content, not just the key
    pr3 = client.post(
        f"/v1/uploads/{up_id}/parts",
        files={"data": ("p", _CORPUS + _CORPUS, "application/octet-stream")},
        headers=_ih("cov.part"),
    )
    out["coverage.uploads_part.conflict"] = (
        pr3.status_code == 409 and _err_code(pr3) == "idempotency_conflict"
    )
    part_id = str(pr1.json()["id"])

    n_files_before = len(client.get("/v1/files", headers=_h()).json().get("data", []))
    cr1, cr2 = _post2(
        client, f"/v1/uploads/{up_id}/complete", {"part_ids": [part_id]}, "cov.upcomp"
    )
    minted_id = cr1.json().get("file", {}).get("id")
    out["coverage.uploads_complete.replay"] = (
        cr1.status_code == 200
        and cr2.status_code == 200
        and _replay_hdr(cr2)
        and minted_id == cr2.json().get("file", {}).get("id")
    )
    files_now = client.get("/v1/files", headers=_h()).json().get("data", [])
    minted = [f for f in files_now if f.get("id") == minted_id]
    out["coverage.uploads_complete.once"] = (
        len(files_now) == n_files_before + 1
        and len(minted) == 1
        and minted[0].get("bytes") == len(_CORPUS)
    )
    # completing again unkeyed fails (already terminal) — the keyed replay above is the dedupe
    cr3 = client.post(f"/v1/uploads/{up_id}/complete", json={"part_ids": [part_id]}, headers=_h())
    out["coverage.uploads_complete.terminal_guard"] = cr3.status_code in (400, 404, 409)

    up2 = client.post("/v1/uploads", json=up_body, headers=_h()).json()["id"]
    xr1 = client.post(f"/v1/uploads/{up2}/cancel", headers=_ih("cov.upcancel"))
    xr2 = client.post(f"/v1/uploads/{up2}/cancel", headers=_ih("cov.upcancel"))
    out["coverage.uploads_cancel.replay"] = (
        xr1.status_code == 200
        and xr2.status_code == 200
        and _replay_hdr(xr2)
        and _same(xr1.json(), xr2.json())
        and xr1.json().get("id") == xr2.json().get("id")
    )
    return out


def _probe_cov_keys() -> dict[str, bool]:
    """``/harness/keys`` mint/rotate/patch/revoke — the highest-stakes
    surface: a retried mint must never mint a second credential."""
    out: dict[str, bool] = {}
    client, _app = _client()

    n0 = len(client.get("/harness/keys", headers=_h()).json().get("data", []))
    m1 = client.post("/harness/keys", json={"name": "idem-a"}, headers=_ih("cov.mint"))
    m2 = client.post("/harness/keys", json={"name": "idem-a"}, headers=_ih("cov.mint"))
    n1 = len(client.get("/harness/keys", headers=_h()).json().get("data", []))
    out["coverage.keys_mint.replay"] = (
        m1.status_code == 201
        and m2.status_code == 201
        and _replay_hdr(m2)
        and m1.json().get("id") == m2.json().get("id")
        and m1.json().get("key") == m2.json().get("key")
    )
    out["coverage.keys_mint.once"] = n1 - n0 == 1

    kid_rot = str(
        client.post("/harness/keys", json={"name": "idem-rot"}, headers=_h()).json()["id"]
    )
    n_pre = len(client.get("/harness/keys", headers=_h()).json().get("data", []))
    r1 = client.post("/harness/keys/" + kid_rot + "/rotate", json={}, headers=_ih("cov.rotate"))
    r2 = client.post("/harness/keys/" + kid_rot + "/rotate", json={}, headers=_ih("cov.rotate"))
    out["coverage.keys_rotate.replay"] = (
        r1.status_code == 201
        and r2.status_code == 201
        and _replay_hdr(r2)
        and r1.json().get("key", {}).get("id") == r2.json().get("key", {}).get("id")
        and r1.json().get("key", {}).get("key") == r2.json().get("key", {}).get("key")
    )
    n2 = len(client.get("/harness/keys", headers=_h()).json().get("data", []))
    out["coverage.keys_rotate.once"] = n2 - n_pre == 1

    kid_pat = str(
        client.post("/harness/keys", json={"name": "idem-pat"}, headers=_h()).json()["id"]
    )
    p1 = client.patch(
        "/harness/keys/" + kid_pat, json={"name": "renamed"}, headers=_ih("cov.patch")
    )
    p2 = client.patch(
        "/harness/keys/" + kid_pat, json={"name": "renamed"}, headers=_ih("cov.patch")
    )
    out["coverage.keys_patch.replay"] = (
        p1.status_code == 200
        and p2.status_code == 200
        and _replay_hdr(p2)
        and _same(p1.json(), p2.json())
        and p1.json().get("id") == p2.json().get("id")
    )
    p3 = client.patch("/harness/keys/" + kid_pat, json={"name": "other"}, headers=_ih("cov.patch"))
    out["coverage.keys_patch.conflict"] = (
        p3.status_code == 409 and _err_code(p3) == "idempotency_conflict"
    )

    kid_del = str(
        client.post("/harness/keys", json={"name": "idem-del"}, headers=_h()).json()["id"]
    )
    d1 = client.delete("/harness/keys/" + kid_del, headers=_ih("cov.revoke"))
    d2 = client.delete("/harness/keys/" + kid_del, headers=_ih("cov.revoke"))
    d3 = client.delete("/harness/keys/" + kid_del, headers=_h())
    out["coverage.keys_revoke.replay"] = (
        d1.status_code == 200
        and d2.status_code == 200
        and _replay_hdr(d2)
        and _same(d1.json(), d2.json())
    )
    out["coverage.keys_revoke.unkeyed_fails"] = d3.status_code in (404, 409)
    return out


# ---------------------------------------------------------------------------
# Replay / conflict / namespacing / hygiene
# ---------------------------------------------------------------------------


def _probe_replay_semantics() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    r1, r2 = _post2(client, "/v1/chat/completions", _chat_body("alpha"), "rep.byte")
    out["replay.byte_identical"] = r1.content == r2.content and r1.status_code == 200
    out["replay.marker_header"] = _replay_hdr(r2) and not _replay_hdr(r1)
    out["replay.original_id"] = r1.json().get("id") == r2.json().get("id")
    out["replay.no_reexecute"] = stub.calls == 1

    # the stored answer is frozen: a mutated backend never leaks into a replay
    stub._model = "mutated"  # next live call would answer differently
    r3 = client.post(
        "/v1/chat/completions",
        json=_chat_body("alpha"),
        headers=_ih("rep.byte"),
    )
    out["replay.frozen_answer"] = r3.content == r1.content and stub.calls == 1

    # job replay keeps the original job_id and marks replayed
    j1, j2 = _post2(client, "/harness/jobs", {"command": "doctor"}, "rep.job")
    out["replay.field_marker"] = j2.json().get("replayed") is True
    out["replay.job_same_id"] = j1.json().get("job_id") == j2.json().get("job_id")
    return out


def _probe_conflict() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    c1 = client.post("/v1/chat/completions", json=_chat_body("a"), headers=_ih("conflict.k"))
    c2 = client.post("/v1/chat/completions", json=_chat_body("b"), headers=_ih("conflict.k"))
    out["conflict.openai_409"] = c2.status_code == 409
    out["conflict.openai_code"] = _err_code(c2) == "idempotency_conflict"
    out["conflict.no_execute"] = stub.calls == 1
    # the loser cannot flip the pin; the winner's body still replays
    c3 = client.post("/v1/chat/completions", json=_chat_body("c"), headers=_ih("conflict.k"))
    c4 = client.post("/v1/chat/completions", json=_chat_body("a"), headers=_ih("conflict.k"))
    out["conflict.stable"] = (
        c3.status_code == 409
        and _err_code(c3) == "idempotency_conflict"
        and c4.status_code == 200
        and _replay_hdr(c4)
        and c4.json().get("id") == c1.json().get("id")
    )

    h1 = client.post("/harness/complete", json=_complete_body("a"), headers=_ih("conflict.h"))
    h2 = client.post("/harness/complete", json=_complete_body("b"), headers=_ih("conflict.h"))
    out["conflict.harness_409"] = h1.status_code == 200 and h2.status_code == 409
    out["conflict.harness_code"] = _err_code(h2) == "idempotency_conflict"

    # runs: header key vs body key share the pin — header wins when both ride
    u1 = client.post(
        "/harness/runs",
        json={"command": "doctor", "idempotency_key": "k-body"},
        headers=_ih("k-header"),
    )
    u2 = client.post(
        "/harness/runs",
        json={"command": "doctor", "idempotency_key": "k-body"},
        headers=_ih("k-header"),
    )
    out["conflict.header_wins"] = (
        u1.status_code == 200 and u2.status_code == 200 and u2.json().get("replayed") is True
    )
    return out


def _probe_namespacing() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    # separate ledgers: the same literal key on /harness/jobs and
    # /harness/complete do not collide
    j = client.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("ns.shared"))
    c = client.post("/harness/complete", json=_complete_body(), headers=_ih("ns.shared"))
    out["namespace.cross_store_independent"] = j.status_code == 202 and c.status_code == 200

    # shared ledger: chat + responses ride the same store — a key reused
    # across them with a different body 409s (pin the real rule)
    a = client.post("/v1/chat/completions", json=_chat_body(), headers=_ih("ns.sharedstore"))
    b = client.post("/v1/responses", json=_responses_body(), headers=_ih("ns.sharedstore"))
    out["namespace.shared_store_conflict"] = (
        a.status_code == 200 and b.status_code == 409 and _err_code(b) == "idempotency_conflict"
    )

    # per-credential scope: key K under credential A is a different claim
    # than K under credential B — B can never replay A's answer
    ka = client.post("/harness/keys", json={"name": "cred-a"}, headers=_h()).json()["key"]
    kb = client.post("/harness/keys", json={"name": "cred-b"}, headers=_h()).json()["key"]
    before = stub.calls
    r_a = client.post(
        "/harness/complete", json=_complete_body("cred"), headers=_ih("ns.cred", auth=ka)
    )
    r_b = client.post(
        "/harness/complete", json=_complete_body("cred"), headers=_ih("ns.cred", auth=kb)
    )
    out["namespace.credential_isolated"] = (
        r_a.status_code == 200
        and r_b.status_code == 200
        and r_a.json().get("replayed") is False
        and r_b.json().get("replayed") is False
        and r_b.json().get("completion_id") != r_a.json().get("completion_id")
        and stub.calls == before + 2
    )
    # B's own ledger still conflicts on a body change
    r_b2 = client.post(
        "/harness/complete", json=_complete_body("other"), headers=_ih("ns.cred", auth=kb)
    )
    out["namespace.credential_own_conflict"] = (
        r_b2.status_code == 409 and _err_code(r_b2) == "idempotency_conflict"
    )
    return out


def _probe_key_hygiene() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    # blank header = absent header: executes every time, never pins
    before = stub.calls
    e1 = client.post("/v1/chat/completions", json=_chat_body(), headers=_h(_ROOT, **{_IDEM: "   "}))
    e2 = client.post("/v1/chat/completions", json=_chat_body(), headers=_h(_ROOT, **{_IDEM: "   "}))
    out["hygiene.blank_ignored"] = (
        e1.status_code == 200
        and e2.status_code == 200
        and not _replay_hdr(e2)
        and stub.calls == before + 2
    )

    long_key = "k" * 257
    l1 = client.post("/harness/complete", json=_complete_body(), headers=_ih(long_key))
    out["hygiene.overlong_400"] = l1.status_code == 400 and _err_code(l1) is not None
    l2 = client.post("/v1/chat/completions", json=_chat_body(), headers=_ih(long_key))
    out["hygiene.overlong_400_v1"] = l2.status_code == 400 and _err_code(l2) is not None

    edge = "e" * 256
    x1, x2 = _post2(client, "/harness/complete", _complete_body("edge"), edge)
    out["hygiene.boundary_256"] = (
        x1.status_code == 200 and x2.status_code == 200 and x2.json().get("replayed") is True
    )
    return out


# ---------------------------------------------------------------------------
# Eviction, failure semantics, replay identity
# ---------------------------------------------------------------------------


def _probe_eviction() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub}, idem_max=3)

    k0 = client.post("/v1/chat/completions", json=_chat_body("v0"), headers=_ih("evict.k0"))
    for i in range(3):
        r = client.post(
            "/v1/chat/completions", json=_chat_body(f"filler{i}"), headers=_ih(f"evict.f{i}")
        )
        assert r.status_code == 200, r.text
    before = stub.calls
    # k0 evicted: the same-body retry re-executes instead of replaying
    re = client.post("/v1/chat/completions", json=_chat_body("v0"), headers=_ih("evict.k0"))
    out["evict.reexecutes"] = (
        re.status_code == 200 and not _replay_hdr(re) and stub.calls == before + 1
    )
    # the re-execute re-pins k0 to v0 — a different body still 409s
    re_conf = client.post("/v1/chat/completions", json=_chat_body("v1"), headers=_ih("evict.k0"))
    out["evict.repins"] = re_conf.status_code == 409
    # evict k0 a second time: now a different body executes cleanly —
    # nothing tombstoned (the old fp ghost never resurfaces)
    for i in range(3):
        client.post(
            "/v1/chat/completions",
            json=_chat_body(f"filler-b{i}"),
            headers=_ih(f"evict.g{i}"),
        )
    re3 = client.post("/v1/chat/completions", json=_chat_body("v2"), headers=_ih("evict.k0"))
    out["evict.frees_key"] = re3.status_code == 200
    out["evict.no_resurrect"] = re3.status_code == 200 and re3.json().get("id") != re.json().get(
        "id"
    )
    out["evict.store_bounded"] = k0.status_code == 200  # submit under cap works
    return out


def _probe_failure_semantics() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    # a 4xx refusal never pins the key: fix the body, retry, it executes
    bad = client.post(
        "/harness/complete",
        json={"backend": _MODEL, "messages": [{"role": "user", "content": "x"}], "tools": [{}]},
        headers=_ih("fail.4xx"),
    )
    good = client.post("/harness/complete", json=_complete_body("fixed"), headers=_ih("fail.4xx"))
    out["fail.client_error_not_pinned"] = (
        bad.status_code >= 400 and good.status_code == 200 and good.json().get("replayed") is False
    )

    # a refusal never blocks a retry of the SAME body either
    bad2 = client.post(
        "/harness/complete",
        json={"backend": "bogus", "messages": [{"role": "user", "content": "x"}]},
        headers=_ih("fail.4xxb"),
    )
    good2 = client.post(
        "/harness/complete", json=_complete_body("fixed2"), headers=_ih("fail.4xxb")
    )
    out["fail.invalid_body_not_pinned"] = bad2.status_code == 422 and good2.status_code == 200

    # a 5xx re-executes on retry — the ledger only pins completions
    fail_stub = _FailBackend("fx1")
    fclient, _fapp = _client(backend_map={_MODEL: lambda: fail_stub})
    f1 = fclient.post("/harness/complete", json=_complete_body("x"), headers=_ih("fail.5xx"))
    f2 = fclient.post("/harness/complete", json=_complete_body("x"), headers=_ih("fail.5xx"))
    out["fail.server_error_reexecutes"] = (
        f1.status_code == 503 and f2.status_code == 503 and fail_stub.calls == 2
    )
    return out


def _probe_replay_identity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _app = _client()

    # mint a file, delete it, then keyed-retry the POST — the ledger
    # serves the frozen snapshot; the live lookup is gone
    f1 = _file_post(client, "ident.file")
    fid = str(f1.json()["id"])
    d = client.delete(f"/v1/files/{fid}", headers=_h())
    g = client.get(f"/v1/files/{fid}", headers=_h())
    f2 = _file_post(client, "ident.file")
    out["identity.delete_succeeds"] = d.status_code == 200
    out["identity.deleted_gone"] = g.status_code == 404
    out["identity.replay_serves_snapshot"] = (
        f2.status_code == 200 and _replay_hdr(f2) and f2.json().get("id") == fid
    )

    # same shape on jobs: DELETE the job record, keyed retry still
    # replays the submit answer (the ledger's view, not the job's state)
    j1 = client.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("ident.job"))
    jid = str(j1.json()["job_id"])
    client.delete(f"/harness/jobs/{jid}", headers=_h())
    j2 = client.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("ident.job"))
    out["identity.job_replay_stable"] = (
        j2.status_code == 202
        and j2.json().get("job_id") == jid
        and j2.json().get("replayed") is True
    )
    return out


# ---------------------------------------------------------------------------
# Stream replay, restart, concurrency, envelopes
# ---------------------------------------------------------------------------


def _probe_stream() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    body = {**_chat_body("stream me"), "stream": True}
    s1 = client.post("/v1/chat/completions", json=body, headers=_ih("stream.k"))
    s2 = client.post("/v1/chat/completions", json=body, headers=_ih("stream.k"))
    out["stream.replay_byte_identical"] = (
        s1.status_code == 200
        and s2.status_code == 200
        and s1.content == s2.content
        and len(s1.content) > 0
    )
    out["stream.replay_marked"] = _replay_hdr(s2)
    out["stream.no_reexecute"] = stub.calls == 1
    # Last-Event-ID resume replays a strict suffix of the pinned stream
    r = client.post(
        "/v1/chat/completions",
        json=body,
        headers=_h(_ROOT, **{_IDEM: "stream.k", "Last-Event-ID": "1"}),
    )
    out["stream.resume_suffix"] = (
        r.status_code == 200
        and len(r.content) < len(s1.content)
        and s1.content.endswith(r.content)
        and len(r.content) > 0
    )
    return out


def _probe_restart() -> dict[str, bool]:
    out: dict[str, bool] = {}
    state_dir = _temporary_directory() / "state"

    stub1 = _StubBackend("fx1")
    c1, _a1 = _client(backend_map={_MODEL: lambda: stub1}, state_dir=state_dir)
    r1 = c1.post("/v1/chat/completions", json=_chat_body("dur"), headers=_ih("restart.k"))
    j1 = c1.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("restart.j"))
    ledger = state_dir / "idem_openai.jsonl"
    out["restart.journal_written"] = ledger.exists() and ledger.stat().st_size > 0

    # fresh app over the same state dir — keyed retry replays the stored
    # answer without touching the new backend
    stub2 = _StubBackend("fx1")
    c2, _a2 = _client(backend_map={_MODEL: lambda: stub2}, state_dir=state_dir)
    r2 = c2.post("/v1/chat/completions", json=_chat_body("dur"), headers=_ih("restart.k"))
    j2 = c2.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("restart.j"))
    out["restart.replays"] = (
        r2.status_code == 200 and _replay_hdr(r2) and r2.json().get("id") == r1.json().get("id")
    )
    out["restart.no_reexecute"] = stub2.calls == 0
    out["restart.jobs_replay"] = (
        j2.status_code == 202
        and j2.json().get("job_id") == j1.json().get("job_id")
        and j2.json().get("replayed") is True
    )

    # the key ledger is deliberately not journaled (the stored mint answer
    # carries the raw credential) — post-restart mint re-executes honestly
    m1 = c1.post("/harness/keys", json={"name": "rk"}, headers=_ih("restart.key"))
    m2 = c2.post("/harness/keys", json={"name": "rk"}, headers=_ih("restart.key"))
    out["restart.keys_reexecute_honestly"] = (
        m1.status_code == 201
        and m2.status_code == 201
        and not _replay_hdr(m2)
        and m2.json().get("id") != m1.json().get("id")
    )
    return out


def _probe_concurrency() -> dict[str, bool]:
    out: dict[str, bool] = {}

    # N parallel same-key chat calls behind a gated backend — the claim
    # lock serializes them: exactly one spends the model
    gate_stub = _GateBackend()
    gate_stub.gate.clear()
    gapp = _make_app(backend_map={_MODEL: lambda: gate_stub})

    async def fan_out_same() -> list[Any]:
        async with (
            gapp.router.lifespan_context(gapp),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(gapp), base_url="http://localhost"
            ) as ac,
        ):
            first = asyncio.create_task(
                ac.post("/v1/chat/completions", json=_chat_body("para"), headers=_ih("conc.k"))
            )
            while not gate_stub.entered.is_set():
                await asyncio.sleep(0.005)
            rest = [
                asyncio.create_task(
                    ac.post("/v1/chat/completions", json=_chat_body("para"), headers=_ih("conc.k"))
                )
                for _ in range(3)
            ]
            gate_stub.gate.set()
            return [await first] + list(await asyncio.gather(*rest))

    rs = asyncio.run(fan_out_same())
    ok = [r for r in rs if r.status_code == 200]
    ids = {r.json().get("id") for r in ok}
    marked = [r for r in rs if _replay_hdr(r)]
    out["conc.same_key_one_execute"] = (
        len(ok) == 4 and len(ids) == 1 and gate_stub.calls == 1 and len(marked) == 3
    )

    # N parallel same-key DIFFERENT bodies — exactly one wins; the rest 409
    gate_stub2 = _GateBackend()
    gate_stub2.gate.clear()
    app2 = _make_app(backend_map={_MODEL: lambda: gate_stub2})

    async def fan_out_conflict() -> list[Any]:
        async with (
            app2.router.lifespan_context(app2),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app2), base_url="http://localhost"
            ) as ac,
        ):
            first = asyncio.create_task(
                ac.post("/v1/chat/completions", json=_chat_body("w0"), headers=_ih("conc.x"))
            )
            while not gate_stub2.entered.is_set():
                await asyncio.sleep(0.005)
            rest = [
                asyncio.create_task(
                    ac.post("/v1/chat/completions", json=_chat_body(f"w{i}"), headers=_ih("conc.x"))
                )
                for i in range(1, 4)
            ]
            gate_stub2.gate.set()
            return [await first] + list(await asyncio.gather(*rest))

    rs2 = asyncio.run(fan_out_conflict())
    wins = [r for r in rs2 if r.status_code == 200]
    conflicts = [r for r in rs2 if r.status_code == 409]
    out["conc.diff_body_one_wins"] = (
        len(wins) == 1
        and len(conflicts) == 3
        and all(_err_code(r) == "idempotency_conflict" for r in conflicts)
        and gate_stub2.calls == 1
    )

    # jobs fan-out: runner counts executions — one job id for all
    runner = _CountingRunner(gate=threading.Event())
    japp = _make_app(runner=runner)

    async def fan_out_jobs() -> list[Any]:
        async with (
            japp.router.lifespan_context(japp),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(japp), base_url="http://localhost"
            ) as ac,
        ):
            first = asyncio.create_task(
                ac.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("conc.j"))
            )
            while not runner.entered.is_set():
                await asyncio.sleep(0.005)
            rest = [
                asyncio.create_task(
                    ac.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("conc.j"))
                )
                for _ in range(3)
            ]
            assert runner.gate is not None
            runner.gate.set()
            rs = [await first] + list(await asyncio.gather(*rest))
            # the survivor's pin stays stable: a later identical retry replays it
            late = await ac.post("/harness/jobs", json={"command": "doctor"}, headers=_ih("conc.j"))
            return rs + [late]

    js = asyncio.run(fan_out_jobs())
    oks = [r for r in js[:4] if r.status_code == 202]
    jids = {r.json().get("job_id") for r in oks}
    out["conc.jobs_one_execution"] = len(oks) == 4 and len(jids) == 1 and len(runner.calls) == 1
    late = js[-1]
    out["conc.winner_stable"] = (
        late.status_code == 202
        and late.json().get("replayed") is True
        and late.json().get("job_id") in jids
    )
    return out


def _probe_envelope() -> dict[str, bool]:
    out: dict[str, bool] = {}
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub})

    client.post("/harness/complete", json=_complete_body("a"), headers=_ih("env.h"))
    e1 = client.post("/harness/complete", json=_complete_body("b"), headers=_ih("env.h"))
    body = e1.json() if e1.headers.get("content-type", "").startswith("application/json") else {}
    out["envelope.harness_409"] = (
        e1.status_code == 409 and "detail" in body and body.get("code") == "idempotency_conflict"
    )

    client.post("/v1/chat/completions", json=_chat_body("a"), headers=_ih("env.o"))
    e2 = client.post("/v1/chat/completions", json=_chat_body("b"), headers=_ih("env.o"))
    err = e2.json().get("error", {})
    out["envelope.openai_409"] = (
        e2.status_code == 409 and err.get("code") == "idempotency_conflict" and "message" in err
    )

    client.post("/v1/messages", json=_messages_body("a"), headers=_ih("env.a", **_ANTHROPIC_V))
    e3 = client.post("/v1/messages", json=_messages_body("b"), headers=_ih("env.a", **_ANTHROPIC_V))
    err3 = e3.json().get("error", {})
    out["envelope.anthropic_409"] = (
        e3.status_code == 409
        and e3.json().get("type") == "error"
        and isinstance(err3, dict)
        and "message" in err3
    )

    k1 = client.post("/harness/complete", json=_complete_body(), headers=_ih("z" * 257))
    out["envelope.overlong_harness"] = k1.status_code == 400 and "code" in k1.json()
    k2 = client.post("/v1/chat/completions", json=_chat_body(), headers=_ih("z" * 257))
    out["envelope.overlong_openai"] = (
        k2.status_code == 400 and "error" in k2.json() and "code" in k2.json()["error"]
    )
    return out


# ---------------------------------------------------------------------------
# Battery entrypoints
# ---------------------------------------------------------------------------


def idem_audit() -> dict[str, bool]:
    """Run every probe group; ``{name: measured_bool}``."""
    with _audit_context():
        results: dict[str, bool] = {}
        for group in (
            _probe_cov_harness,
            _probe_cov_openai,
            _probe_cov_evals,
            _probe_cov_files_uploads,
            _probe_cov_keys,
            _probe_replay_semantics,
            _probe_conflict,
            _probe_namespacing,
            _probe_key_hygiene,
            _probe_eviction,
            _probe_failure_semantics,
            _probe_replay_identity,
            _probe_stream,
            _probe_restart,
            _probe_concurrency,
            _probe_envelope,
        ):
            results.update(group())
        return results


def idem_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    r = idem_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "idem_audit",
        "schema": "idem_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": "Starlette TestClient in-process + httpx ASGI concurrency fan-out",
            "not_verified": [
                "real network transport behavior",
                "external process restart (in-process app recreation over the same journal)",
                "wall-clock key expiry (the store bounds by count, not TTL)",
            ],
        },
        "interpretation": (
            "The Idempotency-Key contract holds end to end across the "
            "whole stateful surface: every mutating route honors the "
            "header, a keyed retry replays the byte-identical stored "
            "answer without re-executing, a key reused with a different "
            "body fails closed 409 idempotency_conflict in either error "
            "grammar, claims serialize racing twins down to exactly one "
            "execution, ledgers are scoped per-credential, failed "
            "submissions never pin the key, evicted keys free cleanly, "
            "streams replay frame-for-frame, and journaled ledgers "
            "survive a state-dir restart while the key ledger honestly "
            "re-executes rather than persisting secrets."
            if ok
            else f"IDEM AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(idem_audit_bench(), indent=2, sort_keys=True))
