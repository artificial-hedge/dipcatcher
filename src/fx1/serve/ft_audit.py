"""ft_audit — ``/v1/fine_tuning/jobs`` deep audit.

Probe battery over the OpenAI-shaped fine-tuning surface end to end:
submit validation (required fields, model trainability, hyperparameter
bounds, suffix/seed typing, extra-field + callback_url refusal), file
binding (purpose, deletion, malformed corpora, validation_file twin
rules), the queued→running→succeeded lifecycle (frozen submit view,
monotone forward transitions, honest terminal records), the ``ft:``
model registry (card minted only on a real checkpoint, resolves through
chat, delete tombstones everywhere, eviction drops the card), the event
feed (oldest-first, bounded, cursors, levels + data verbatim), the
checkpoints surface (honest empty mid-run, one card-derived entry
post-success, ``step_number``/``metrics`` never fabricated),
cooperative cancel + pause/resume (queued/running/paused outcomes,
terminal 409s, drain-gated resume), concurrency (parallel mints,
credential-scoped Idempotency-Key dedup, shared-file policy), the
admission gate ordering (auth → validation → drain → over_capacity),
``--state-dir`` journaled durability (terminal as-was, non-terminal
fails closed, idem keys survive, deletes never resurrect, parked jobs
reported honestly), metrics honesty (``trained_tokens`` and artifacts
echo exactly what the runner produced — never invented), envelope
uniformity (every refusal in the OpenAI ``{error}`` grammar), metering
(every authenticated call bills ``uses`` — refusals included — while
``tokens``/``served`` never move for job control-plane work), the
terminal webhook (signed delivery, retry policy, no post-restart
resend), the executor-admission edge (a refused hand-off leaves no
ghost job and no phantom idem claim), and the ``HarnessClient`` wire
legs + error map.

Found while building this lane (fixed in the same commit):

* ``create_finetune_job`` returned the live ``job`` object after
  ``jobs_executor.submit``: the worker mutates ``entry.job`` in place
  and response serialization happens after return, so a fast (or slow
  serializer) window let the submit body leak ``running``/``succeeded``
  status, ``fine_tuned_model``, and ``result_files`` — a torn submit
  view, the same defect class the race lane fixed on the other five
  async surfaces. The emitted record is now frozen pre-dispatch via a
  ``model_validate`` of the job's JSON projection (``deepcopy`` cannot
  be used — the record carries a ``threading.Lock`` private attr that
  cannot be pickled).
* A ``RuntimeError`` from ``jobs_executor.submit`` (executor gone —
  shutdown race) left the freshly ``put`` job in the store as
  ``queued`` forever, journaled, idem-claimed, and replayable — a ghost
  job whose keyed retry answered the phantom record instead of
  executing. The submit path now tombstones the record via the new
  ``FTJobStore.delete`` (``{"deleted": job_id}`` journal op, replay
  drops it) — the same convention the jobs lane applied to
  ``_JobStore.delete``.

Sealed ``ft_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from quant_fund.research.receipt_v2 import git_revision

__all__ = ["ft_audit", "ft_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_IDEM = "Idempotency-Key"
_FT = "/v1/fine_tuning/jobs"
_FILES = "/v1/files"
_MODELS = "/v1/models"
_CHAT = "/v1/chat/completions"
_DRAIN = "/harness/drain"
_KEYS = "/harness/keys"
_TERMINAL = {"succeeded", "failed", "cancelled"}
_RANK = {"queued": 0, "paused": 0, "running": 1, "succeeded": 2, "failed": 2, "cancelled": 2}
_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_CORPUS2 = (
    _CORPUS + b'{"messages":[{"role":"user","content":"q2"},{"role":"assistant","content":"a2"}]}\n'
)


# ---------------------------------------------------------------------------
# Stub plumbing — runners, app/client factories, poll helpers
# ---------------------------------------------------------------------------


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _ok_runner(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
    """Runner that produces the honest full outcome — model name, a real
    checkpoint dir, one artifact file, and a measured token count."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    emit("info", "runner working", {"stage": "train"})
    ckpt = spec.work_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "weights.bin").write_bytes(b"w")
    art = spec.work_dir / "receipt.json"
    art.write_text('{"ok": true}')
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        artifacts={"receipt.json": art},
        trained_tokens=1234,
        checkpoint=str(ckpt),
    )


def _quiet_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Returns instantly — exercises the submit-view freeze: a fast
    runner flips the record in-place before the response serializes."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    return FTJobOutcome(fine_tuned_model=spec.ft_model_name)


def _fail_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    raise RuntimeError("synthetic runner fault")


def _no_ckpt_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Reports a model name with no checkpoint — the card must not mint
    (a registered ``ft:`` name must resolve to servable weights)."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    return FTJobOutcome(fine_tuned_model=spec.ft_model_name, checkpoint=None)


def _none_tokens_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Honest ``None`` metrics — the wire must show nulls, not zeros."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        checkpoint=str(spec.work_dir),
    )


def _levels_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Emits every event level + a data payload — the feed must carry
    them verbatim."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    emit("info", "i-msg", {"k": 1})
    emit("warn", "w-msg", None)
    emit("error", "e-msg", {"why": "synthetic"})
    return FTJobOutcome(fine_tuned_model=None)


def _many_events_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Pushes the feed past ``_FT_EVENT_CAP`` — the bound must hold."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    for i in range(300):
        emit("info", f"tick {i}", None)
    return FTJobOutcome(fine_tuned_model=None)


class _GateRunner:
    """Parks inside the runner until released — occupies its inflight
    slot and mid-pipeline pause/cancel gates deterministically. The
    release is registered on the audit resource stack so teardown never
    strands a worker thread."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[Any] = []
        self.parked = threading.Event()

    def __call__(self, spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

        self.calls.append(spec)
        deadline = time.monotonic() + 30.0
        while not self.release.is_set():
            if should_cancel():
                return FTJobOutcome(fine_tuned_model=None)
            if pause_gate():  # returns True iff a cancel landed parked
                return FTJobOutcome(fine_tuned_model=None)
            if time.monotonic() > deadline:
                raise AssertionError("gate never released")
            time.sleep(0.01)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name)


def _gated(gate: _GateRunner) -> None:
    """Register ``gate.release`` on the audit stack — no probe may leave
    a worker parked past teardown."""
    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

    _RESOURCES.get().callback(gate.release.set)


def _make_app(
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    ft_runner: Any | None = None,
    backend: Any | None = None,
    **create_kw: Any,
) -> FastAPI:
    """``create_app`` under the ambient env (``_audit_context`` swept
    ``FX1_*``); ``backend`` installs one stub for every link."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _StubBackend, _temporary_directory  # noqa: PLC0415

    stub = backend if backend is not None else _StubBackend()
    create_kw.setdefault("ft_dir", _temporary_directory() / "ft")
    return api_mod.create_app(
        harness=Harness(runner=runner or _fast_runner),
        backend_resolver=lambda *a, **k: stub,
        ft_runner=ft_runner or _ok_runner,
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
    """Flat ``code`` or nested OpenAI ``error.code``."""
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


def _err_type(r: Any) -> str | None:
    if not r.headers.get("content-type", "").startswith("application/json"):
        return None
    err = r.json().get("error")
    if isinstance(err, dict) and isinstance(err.get("type"), str):
        return str(err["type"])
    return None


def _upload(client: TestClient, content: bytes = _CORPUS, purpose: str = "fine-tune") -> str:
    r = client.post(
        _FILES,
        files={"file": ("train.jsonl", content)},
        data={"purpose": purpose},
        headers=_h(),
    )
    assert r.status_code == 200, f"upload refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _submit(client: TestClient, fid: str, **fields: Any) -> Any:
    body = {"model": "fx1", "training_file": fid}
    body.update(fields)
    return client.post(_FT, json=body, headers=_h())


def _wait_terminal(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        j = client.get(f"{_FT}/{job_id}", headers=_h())
        body = j.json()
        if j.status_code == 200 and body.get("status") in _TERMINAL:
            return dict(body)
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never reached terminal: {body}")
        time.sleep(0.02)


def _wait_status(client: TestClient, job_id: str, status: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if client.get(f"{_FT}/{job_id}", headers=_h()).json().get("status") == status:
            return
        time.sleep(0.01)


def _status_seq(client: TestClient, job_id: str, timeout_s: float = 30.0) -> list[str]:
    """Poll until terminal; return the observed status sequence."""
    seq: list[str] = []
    deadline = time.monotonic() + timeout_s
    while True:
        st = client.get(f"{_FT}/{job_id}", headers=_h()).json().get("status")
        if not seq or seq[-1] != st:
            seq.append(str(st))
        if st in _TERMINAL or time.monotonic() > deadline:
            return seq
        time.sleep(0.01)


def _events(client: TestClient, job_id: str, **q: Any) -> dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in q.items() if v is not None)
    sep = "?" if qs else ""
    return dict(client.get(f"{_FT}/{job_id}/events{sep}{qs}", headers=_h()).json())


def _mint(client: TestClient, root: str, **fields: Any) -> dict[str, Any]:
    r = client.post(_KEYS, json=fields, headers=_h(root))
    assert r.status_code == 201, f"key mint refused: {r.status_code} {r.text}"
    return dict(r.json())


def _key_card(client: TestClient, root: str, key_id: str) -> dict[str, Any]:
    return dict(client.get(f"{_KEYS}/{key_id}", headers=_h(root)).json())


def _usage_card(client: TestClient, root: str, key_id: str) -> dict[str, Any]:
    """``/harness/keys/{id}/usage`` — the metered card with ``served``."""
    r = client.get(f"{_KEYS}/{key_id}/usage", headers=_h(root))
    assert r.status_code == 200, r.text
    return dict(r.json())


class _HookSink:
    """Webhook sink — records (headers, body, path) and answers a fixed
    status per path: ``/reject`` 404 (definitive), ``/flaky`` 500
    (transient), else 200."""

    def __init__(self) -> None:
        self.hits: list[tuple[str, dict[str, str], bytes]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — stdlib handler API
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.hits.append((self.path, dict(self.headers), body))
                if self.path == "/reject":
                    self.send_response(404)
                elif self.path == "/flaky":
                    self.send_response(500)
                else:
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                return

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

        _RESOURCES.get().callback(self._srv.shutdown)
        _RESOURCES.get().callback(self._srv.server_close)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def wait_hits(self, n: int, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s
        while len(self.hits) < n and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(self.hits) >= n, f"webhook sink saw {len(self.hits)} < {n}"

    def wait_verdict(
        self, client: TestClient, job_id: str, timeout_s: float = 15.0
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            job = _wait_terminal(client, job_id)
            if job.get("callback_status") is not None:
                return job
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} never recorded a callback verdict")


# ---------------------------------------------------------------------------
# Section 1 — submit validation
# ---------------------------------------------------------------------------


def _submit_validation_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # required fields — pydantic refuses before the handler
    r = client.post(_FT, json={"model": "fx1"}, headers=_h())
    out["missing_training_file_422"] = r.status_code == 422 and _err_code(r) == "validation"
    r = client.post(_FT, json={"training_file": fid}, headers=_h())
    out["missing_model_422"] = r.status_code == 422 and _err_code(r) == "validation"
    r = client.post(_FT, json={}, headers=_h())
    out["empty_body_422"] = r.status_code == 422

    # model trainability — unknown and known-but-not-trainable both refuse
    r = _submit(client, fid, model="no-such-model")
    out["unknown_model_400"] = r.status_code == 400 and _err_code(r) == "model_not_trainable"
    r = _submit(client, fid, model="byok")
    out["untrainable_link_400"] = r.status_code == 400 and _err_code(r) == "model_not_trainable"
    r = _submit(client, fid, model="local_fx1")
    out["local_fx1_trainable"] = r.status_code == 200

    # bogus file id
    r = _submit(client, "file-nope")
    out["bogus_file_404"] = r.status_code == 404 and _err_code(r) == "file_not_found"

    # hyperparameter bounds — each wall refuses
    for val, name in (
        ({"n_epochs": 0}, "epochs_low"),
        ({"n_epochs": 51}, "epochs_high"),
        ({"batch_size": 0}, "batch_low"),
        ({"batch_size": 513}, "batch_high"),
        ({"learning_rate_multiplier": 0}, "lr_zero"),
        ({"learning_rate_multiplier": 10.5}, "lr_high"),
        ({"learning_rate_multiplier": -1}, "lr_neg"),
    ):
        r = _submit(client, fid, hyperparameters=val)
        out[f"hyper_{name}_422"] = r.status_code == 422 and _err_code(r) == "validation"
    r = _submit(
        client,
        fid,
        hyperparameters={"n_epochs": 2, "batch_size": 4, "learning_rate_multiplier": 2.0},
    )
    out["hyper_valid_ok"] = r.status_code == 200
    job_hp = r.json()
    out["hyper_echoes"] = job_hp.get("hyperparameters", {}).get("n_epochs") == 2

    # suffix typing — pattern + length
    r = _submit(client, fid, suffix="Bad Suffix!")
    out["suffix_pattern_422"] = r.status_code == 422
    r = _submit(client, fid, suffix="x" * 65)
    out["suffix_len_422"] = r.status_code == 422
    r = _submit(client, fid, suffix="ok-name_1")
    out["suffix_valid_ok"] = r.status_code == 200
    out["suffix_echoes"] = r.json().get("user_provided_suffix") == "ok-name_1"

    # seed typing — negative and non-int refuse
    r = _submit(client, fid, seed=-1)
    out["seed_neg_422"] = r.status_code == 422
    r = _submit(client, fid, seed="abc")
    out["seed_str_422"] = r.status_code == 422
    r = _submit(client, fid, seed=7)
    out["seed_valid_ok"] = r.status_code == 200 and r.json().get("seed") == 7

    # method — only "supervised"
    r = _submit(client, fid, method="dpo")
    out["method_dpo_422"] = r.status_code == 422
    r = _submit(client, fid, method="supervised")
    out["method_supervised_ok"] = r.status_code == 200
    out["method_wire_shape"] = r.json().get("method") == {"type": "supervised"}

    # extra fields — OpenAI's integrations/webhook_url are not fields here;
    # extra=forbid refuses them (the fx1 webhook is callback_url/secret)
    r = _submit(client, fid, integrations=[{"type": "wandb"}])
    out["integrations_extra_422"] = r.status_code == 422
    r = _submit(client, fid, webhook_url="https://x.example/hook")
    out["webhook_url_extra_422"] = r.status_code == 422
    r = _submit(client, fid, totally_made_up=True)
    out["unknown_field_422"] = r.status_code == 422

    # callback_url validation — http(s) only, no userinfo, secret needs url
    r = _submit(client, fid, callback_url="ftp://x.example/hook")
    out["callback_scheme_422"] = r.status_code == 422
    r = _submit(client, fid, callback_url="https://u:p@x.example/hook")
    out["callback_userinfo_422"] = r.status_code == 422
    r = _submit(client, fid, callback_url="not a url")
    out["callback_garbage_422"] = r.status_code == 422
    r = _submit(client, fid, callback_secret="sekrit")
    out["callback_secret_alone_422"] = r.status_code == 422
    r = _submit(client, fid, callback_url="http://127.0.0.1:9/hook", callback_secret="s")
    out["callback_pair_ok"] = r.status_code == 200

    # metadata bounds
    r = _submit(client, fid, metadata={f"k{i}": "v" for i in range(17)})
    out["metadata_cap_422"] = r.status_code == 422
    r = _submit(client, fid, metadata={"k": "v"})
    out["metadata_echoes"] = r.status_code == 200 and r.json().get("metadata") == {"k": "v"}

    # Idempotency-Key header validation happens in the claim dep —
    # before any store work
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("k" * 300),
    )
    out["idem_key_len_400"] = r.status_code == 400
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("bad\x01key"),
    )
    out["idem_key_ctrl_400"] = r.status_code == 400
    return out


# ---------------------------------------------------------------------------
# Section 2 — file binding
# ---------------------------------------------------------------------------


def _file_binding_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # purpose: only 'fine-tune' uploads train — a batch file refuses
    bfid = _upload(client, purpose="batch")
    r = _submit(client, bfid)
    out["batch_purpose_400"] = r.status_code == 400 and _err_code(r) == "invalid_training_file"
    r = _submit(client, fid, validation_file=bfid)
    out["batch_purpose_val_400"] = r.status_code == 400 and _err_code(r) == "invalid_training_file"

    # a deleted file is a refusal, not a hang
    dead = _upload(client)
    d = client.delete(f"{_FILES}/{dead}", headers=_h())
    out["file_delete_ok"] = d.status_code == 200 and d.json().get("deleted") is True
    r = _submit(client, dead)
    out["deleted_file_404"] = r.status_code == 404 and _err_code(r) == "file_not_found"
    r = _submit(client, fid, validation_file=dead)
    out["deleted_val_404"] = r.status_code == 404 and _err_code(r) == "file_not_found"

    # malformed corpora refuse at submit — never reach the queue
    bad = {
        "not_json": _upload(client, content=b"not json\n"),
        "no_messages": _upload(client, content=b'{"nope": true}\n'),
        "empty_file": _upload(client, content=b"\n\n"),
        "bad_role": _upload(
            client,
            content=b'{"messages":[{"role":"villain","content":"x"}]}\n',
        ),
        "empty_content": _upload(
            client,
            content=b'{"messages":[{"role":"user","content":""}]}\n',
        ),
        "not_utf8": _upload(client, content=b"\xff\xfe\n"),
    }
    for name, bid in bad.items():
        r = _submit(client, bid)
        out[f"corpus_{name}_400"] = r.status_code == 400 and _err_code(r) == "invalid_training_file"
    r = _submit(client, fid, validation_file=bad["not_json"])
    out["corpus_bad_val_400"] = r.status_code == 400 and _err_code(r) == "invalid_training_file"

    # validation_file mirrors training_file rules; bogus id 404s
    r = _submit(client, fid, validation_file="file-nope")
    out["bogus_val_404"] = r.status_code == 404 and _err_code(r) == "file_not_found"
    r = _submit(client, fid, validation_file=fid)
    out["val_self_ok"] = r.status_code == 200

    # multi-line corpus validates and reports the count in the feed
    fid2 = _upload(client, content=_CORPUS2)
    j = _submit(client, fid2).json()
    ev = _events(client, j["id"])["data"]
    out["validated_event_reports_count"] = any(
        "2 examples" in str(e.get("message", "")) for e in ev
    )
    return out


# ---------------------------------------------------------------------------
# Section 3 — lifecycle
# ---------------------------------------------------------------------------


def _lifecycle_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # submit view is the frozen pre-dispatch record: status queued,
    # no post-start fields — even though this runner finishes instantly
    fast = _client()
    ffid = _upload(fast)
    r0 = _submit(fast, ffid)
    b0 = r0.json()
    out["submit_view_frozen_queued"] = (
        r0.status_code == 200
        and b0.get("status") == "queued"
        and b0.get("fine_tuned_model") is None
        and b0.get("result_files") == []
        and b0.get("finished_at") is None
    )
    _wait_terminal(fast, b0["id"])

    # the wire contract fields
    out["job_object"] = b0.get("object") == "fine_tuning.job"
    out["job_id_shape"] = str(b0.get("id", "")).startswith("ftjob-")
    out["job_org"] = b0.get("organization_id") == "fx1-harness"
    out["job_training_file_echo"] = b0.get("training_file") == ffid
    out["job_validation_null"] = b0.get("validation_file") is None
    out["job_integrations_empty"] = b0.get("integrations") == []
    out["job_estimated_finish_null"] = b0.get("estimated_finish") is None
    out["job_seed_null"] = b0.get("seed") is None
    out["job_error_null"] = b0.get("error") is None

    # lifecycle — a slow runner lets the queued hop be observed;
    # statuses only move forward
    gate = _GateRunner()
    _gated(gate)
    slow = _client(ft_runner=gate)
    sfid = _upload(slow)
    s0 = _submit(slow, sfid, suffix="lc").json()
    seq: list[str] = [str(s0["status"])]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        st = slow.get(f"{_FT}/{s0['id']}", headers=_h()).json().get("status")
        if st != seq[-1]:
            seq.append(str(st))
        if st == "running" or st in _TERMINAL:
            break
        time.sleep(0.01)
    gate.release.set()
    fin = _wait_terminal(slow, s0["id"])
    if fin["status"] not in seq:
        seq.append(str(fin["status"]))
    out["lifecycle_forward"] = (
        all(_RANK[seq[i]] <= _RANK[seq[i + 1]] for i in range(len(seq) - 1)) and seq[0] == "queued"
    )
    out["lifecycle_terminal_succeeded"] = fin["status"] == "succeeded"
    out["finished_stamped"] = (
        isinstance(fin["finished_at"], int) and fin["finished_at"] >= fin["created_at"]
    )
    out["succeeded_ft_name"] = (
        fin["fine_tuned_model"] == f"ft:fx1:lc:{s0['id'].split('-', 1)[1][:12]}"
    )
    out["succeeded_error_null"] = fin["error"] is None

    # default suffix is 'job'
    d0 = _submit(client, fid).json()
    dout = _wait_terminal(client, d0["id"])
    out["default_suffix_job"] = (
        dout["fine_tuned_model"] == f"ft:fx1:job:{d0['id'].split('-', 1)[1][:12]}"
    )

    # result files — registered as real downloadable fine-tune-result files
    rf = dout.get("result_files") or []
    card = client.get(f"{_FILES}/{rf[0]}", headers=_h()).json() if rf else {}
    blob = client.get(f"{_FILES}/{rf[0]}/content", headers=_h()) if rf else None
    out["result_file_registered"] = len(rf) == 1 and card.get("purpose") == "fine-tune-result"
    out["result_file_content_real"] = (
        blob is not None and blob.status_code == 200 and blob.content == b'{"ok": true}'
    )
    out["trained_tokens_reported"] = dout.get("trained_tokens") == 1234

    # a runner fault lands honestly on the record
    fc = _client(ft_runner=_fail_runner)
    ffid2 = _upload(fc)
    f0 = _submit(fc, ffid2).json()
    fout = _wait_terminal(fc, f0["id"])
    out["failed_status"] = fout["status"] == "failed"
    out["failed_error_honest"] = (fout.get("error") or {}).get(
        "code"
    ) == "job_failed" and "synthetic runner fault" in str(
        (fout.get("error") or {}).get("message", "")
    )
    out["failed_no_model"] = fout["fine_tuned_model"] is None
    out["failed_finished_stamped"] = fout["finished_at"] is not None
    fev = _events(fc, f0["id"])["data"]
    out["failed_event_recorded"] = any(
        e.get("level") == "error" and "job failed" in str(e.get("message", "")) for e in fev
    )

    # list surface — newest-first, cursor pagination, limit bounds
    a = _submit(client, fid).json()["id"]
    b = _submit(client, fid).json()["id"]
    c = _submit(client, fid).json()["id"]
    lst = client.get(_FT, headers=_h()).json()
    ids = [j["id"] for j in lst["data"]]
    pos = [i for i in ids if i in {a, b, c}]
    out["list_newest_first"] = pos == [c, b, a]
    for x in (a, b, c):
        _wait_terminal(client, x)
    page1 = client.get(f"{_FT}?limit=1", headers=_h()).json()
    out["list_limit1_has_more"] = page1["has_more"] is True and len(page1["data"]) == 1
    page2 = client.get(f"{_FT}?limit=10&after={page1['data'][0]['id']}", headers=_h()).json()
    out["list_after_cursor"] = (
        all(j["id"] != page1["data"][0]["id"] for j in page2["data"])
        and len(page2["data"]) == len(ids) - 1
    )
    r = client.get(f"{_FT}?limit=0", headers=_h())
    out["list_limit0_422"] = r.status_code == 422
    r = client.get(f"{_FT}?limit=101", headers=_h())
    out["list_limit101_422"] = r.status_code == 422
    r = client.get(f"{_FT}?limit=100", headers=_h())
    out["list_limit100_ok"] = r.status_code == 200
    out["list_object"] = lst.get("object") == "list"

    # unknown job ids refuse everywhere uniformly
    for verb, url in (
        ("get", f"{_FT}/ftjob-nope"),
        ("get", f"{_FT}/ftjob-nope/events"),
        ("get", f"{_FT}/ftjob-nope/checkpoints"),
        ("post", f"{_FT}/ftjob-nope/cancel"),
        ("post", f"{_FT}/ftjob-nope/pause"),
        ("post", f"{_FT}/ftjob-nope/resume"),
    ):
        r = getattr(client, verb)(url, headers=_h())
        out[f"unknown_{url.rsplit('/', 1)[-1]}_404"] = (
            r.status_code == 404 and _err_code(r) == "job_not_found"
        )
    return out


# ---------------------------------------------------------------------------
# Section 4 — ft: model registry
# ---------------------------------------------------------------------------


def _registry_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    calls: list[tuple[str, Any]] = []

    def spy(name: str, *a: Any, **k: Any) -> Any:
        calls.append((name, k.get("checkpoint_dir") or (a[0] if a else None)))
        from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415

        return _StubBackend()

    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _RESOURCES, _temporary_directory  # noqa: PLC0415

    app = api_mod.create_app(
        harness=Harness(runner=_fast_runner),
        backend_resolver=spy,
        ft_runner=_ok_runner,
        ft_dir=_temporary_directory() / "ft",
    )
    client = TestClient(app, raise_server_exceptions=False)
    _RESOURCES.get().callback(app.state.jobs_executor.shutdown, False, cancel_futures=True)

    fid = _upload(client)
    j = _submit(client, fid, suffix="reg").json()
    fin = _wait_terminal(client, j["id"])
    name = fin["fine_tuned_model"]

    models = client.get(_MODELS, headers=_h()).json()["data"]
    out["registry_listed"] = name in {m["id"] for m in models}
    card = client.get(f"{_MODELS}/{name}", headers=_h())
    out["registry_card"] = card.status_code == 200 and card.json().get("id") == name
    out["registry_card_created"] = int(card.json().get("created", 0)) == int(fin["finished_at"])

    # chat resolution — ft: routes to local_fx1 at the registered ckpt
    resp = client.post(
        _CHAT,
        json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
        headers=_h(),
    )
    out["registry_resolves_chat"] = (
        resp.status_code == 200
        and resp.json().get("system_fingerprint") == "local_fx1"
        and calls[-1][0] == "local_fx1"
        and str(calls[-1][1]).endswith("ckpt")
    )
    ghost = client.post(
        _CHAT,
        json={
            "model": "ft:fx1:ghost:000000000000",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=_h(),
    )
    out["registry_ghost_404"] = ghost.status_code == 404 and _err_code(ghost) == "model_not_found"

    # the responses surface resolves the same registry
    resp2 = client.post(
        "/v1/responses",
        json={"model": name, "input": "hi"},
        headers=_h(),
    )
    out["registry_resolves_responses"] = resp2.status_code == 200

    # no-checkpoint outcomes never mint a card
    nc = _client(ft_runner=_no_ckpt_runner)
    nfid = _upload(nc)
    nj = _submit(nc, nfid).json()
    nfin = _wait_terminal(nc, nj["id"])
    out["no_ckpt_not_registered"] = (
        nfin["status"] == "succeeded"
        and nfin["fine_tuned_model"] is not None
        and nc.get(f"{_MODELS}/{nfin['fine_tuned_model']}", headers=_h()).status_code == 404
        and nc.get(f"{_FT}/{nj['id']}/checkpoints", headers=_h()).json()["data"] == []
    )

    # delete — tombstones the card everywhere
    dele = client.delete(f"{_MODELS}/{name}", headers=_h())
    out["registry_delete_ok"] = dele.status_code == 200 and dele.json().get("deleted") is True
    out["registry_delete_card_404"] = (
        client.get(f"{_MODELS}/{name}", headers=_h()).status_code == 404
    )
    models2 = client.get(_MODELS, headers=_h()).json()["data"]
    out["registry_delete_unlisted"] = name not in {m["id"] for m in models2}
    dead = client.post(
        _CHAT,
        json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
        headers=_h(),
    )
    out["registry_delete_unresolves"] = (
        dead.status_code == 404 and _err_code(dead) == "model_not_found"
    )
    ckpts = client.get(f"{_FT}/{j['id']}/checkpoints", headers=_h()).json()
    out["registry_delete_drops_checkpoints"] = ckpts["data"] == []

    # delete rules — builtins and unknowns refuse
    r = client.delete(f"{_MODELS}/fx1", headers=_h())
    out["builtin_delete_400"] = r.status_code == 400 and _err_code(r) == "invalid_request"
    r = client.delete(f"{_MODELS}/ft:fx1:never:000000000000", headers=_h())
    out["unknown_delete_404"] = r.status_code == 404 and _err_code(r) == "model_not_found"

    # evicted job drops its card — bounded store honesty
    evict_client = _client(job_max=1)
    ef1 = _upload(evict_client)
    e1 = _submit(evict_client, ef1, suffix="e1").json()
    efin1 = _wait_terminal(evict_client, e1["id"])
    e2 = _submit(evict_client, ef1, suffix="e2").json()
    _wait_terminal(evict_client, e2["id"])
    out["evicted_job_404"] = evict_client.get(f"{_FT}/{e1['id']}", headers=_h()).status_code == 404
    out["evicted_card_dropped"] = (
        evict_client.get(f"{_MODELS}/{efin1['fine_tuned_model']}", headers=_h()).status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Section 5 — event feed
# ---------------------------------------------------------------------------


def _events_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # a fresh queued job already carries the submit-time validation
    # event — the feed is never fabricated-empty while work is real
    gate = _GateRunner()
    _gated(gate)
    gclient = _client(ft_runner=gate, max_inflight=2)
    gfid = _upload(gclient)
    g0 = _submit(gclient, gfid, hyperparameters={"batch_size": 8}).json()
    ev0 = _events(gclient, g0["id"])
    msgs0 = [str(e.get("message")) for e in ev0["data"]]
    out["fresh_job_has_validate_event"] = any("training file validated" in m for m in msgs0)
    out["batch_size_advisory_event"] = any("batch_size is advisory" in m for m in msgs0)
    out["fresh_event_fields"] = all(
        str(e.get("id", "")).startswith("ftev-")
        and e.get("level") in ("info", "warn", "error")
        and isinstance(e.get("created_at"), int)
        for e in ev0["data"]
    )
    gate.release.set()
    _wait_terminal(gclient, g0["id"])

    # the lifecycle stream grows and persists post-terminal
    j = _submit(client, fid).json()
    _wait_terminal(client, j["id"])
    ev = _events(client, j["id"])
    msgs = [str(e.get("message")) for e in ev["data"]]
    out["feed_grew"] = len(ev["data"]) >= 4  # validated + started + registered + succeeded
    out["feed_oldest_first"] = ev["data"][0]["created_at"] <= ev["data"][-1]["created_at"]
    out["feed_job_started"] = "job started" in msgs
    out["feed_model_registered"] = any(m.startswith("model registered: ") for m in msgs)
    out["feed_job_succeeded_last"] = msgs[-1] == "job succeeded"
    out["feed_ids_unique"] = len({e["id"] for e in ev["data"]}) == len(ev["data"])
    n_ev = len(ev["data"])
    time.sleep(0.1)
    out["feed_stable_post_terminal"] = len(_events(client, j["id"])["data"]) == n_ev
    out["feed_object"] = ev.get("object") == "list" and ev.get("has_more") is False

    # cursor + limit semantics
    page = _events(client, j["id"], limit=1)
    out["feed_limit1_more"] = page["has_more"] is True and len(page["data"]) == 1
    page2 = _events(client, j["id"], limit=50, after=page["data"][0]["id"])
    out["feed_after_continues"] = (
        page2["data"][0]["id"] != page["data"][0]["id"]
        and len(page2["data"]) == n_ev - 1
        and page2["has_more"] is False
    )
    out["feed_bogus_after_empty"] = _events(client, j["id"], after="ftev-nope")["data"] == []
    r = client.get(f"{_FT}/{j['id']}/events?limit=0", headers=_h())
    out["feed_limit0_422"] = r.status_code == 422
    r = client.get(f"{_FT}/{j['id']}/events?limit=101", headers=_h())
    out["feed_limit101_422"] = r.status_code == 422

    # levels + data payloads pass through verbatim
    lv = _client(ft_runner=_levels_runner)
    lfid = _upload(lv)
    l0 = _submit(lv, lfid).json()
    _wait_terminal(lv, l0["id"])
    lev = _events(lv, l0["id"])["data"]
    by_msg = {e["message"]: e for e in lev}
    out["feed_levels_verbatim"] = (
        by_msg["i-msg"]["level"] == "info"
        and by_msg["i-msg"]["data"] == {"k": 1}
        and by_msg["w-msg"]["level"] == "warn"
        and by_msg["w-msg"]["data"] is None
        and by_msg["e-msg"]["level"] == "error"
        and by_msg["e-msg"]["data"] == {"why": "synthetic"}
    )

    # the feed is bounded — past the cap the oldest events drop honestly
    mc = _client(ft_runner=_many_events_runner)
    mfid = _upload(mc)
    m0 = _submit(mc, mfid).json()
    _wait_terminal(mc, m0["id"])
    n_total = 0
    cursor = None
    has_more = False
    for _ in range(4):
        page = _events(mc, m0["id"], limit=100, after=cursor)
        n_total += len(page["data"])
        has_more = page["has_more"]
        if not has_more:
            break
        cursor = page["data"][-1]["id"]
    out["feed_bounded_at_cap"] = has_more is False and n_total == 256
    return out


# ---------------------------------------------------------------------------
# Section 6 — checkpoints
# ---------------------------------------------------------------------------


def _checkpoints_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # honest empty while the job is in-flight
    gate = _GateRunner()
    _gated(gate)
    gclient = _client(ft_runner=gate)
    gfid = _upload(gclient)
    g0 = _submit(gclient, gfid).json()
    _wait_status(gclient, g0["id"], "running")
    mid = gclient.get(f"{_FT}/{g0['id']}/checkpoints", headers=_h()).json()
    out["midrun_empty"] = mid["data"] == [] and mid["first_id"] is None and mid["last_id"] is None
    gate.release.set()
    _wait_terminal(gclient, g0["id"])

    # post-success: exactly one entry derived from the card
    j = _submit(client, fid).json()
    gfin = _wait_terminal(client, j["id"])
    post = client.get(f"{_FT}/{j['id']}/checkpoints", headers=_h()).json()
    items = post["data"]
    out["post_success_one"] = len(items) == 1 and post["has_more"] is False
    if items:
        c0 = items[0]
        out["ckpt_object"] = c0["object"] == "fine_tuning.job.checkpoint"
        out["ckpt_id_shape"] = str(c0["id"]).startswith("ftckpt-")
        out["ckpt_points_at_model"] = c0["fine_tuned_model_checkpoint"] == gfin["fine_tuned_model"]
        out["ckpt_created_real"] = int(c0["created_at"]) == int(gfin["finished_at"])
        out["ckpt_first_last"] = post["first_id"] == c0["id"] and post["last_id"] == c0["id"]
        # honest: no fabricated step metrics
        out["ckpt_no_fabricated_metrics"] = c0["step_number"] is None and c0["metrics"] == {}

    # failed job: no checkpoint
    fc = _client(ft_runner=_fail_runner)
    ffid = _upload(fc)
    f0 = _submit(fc, ffid).json()
    _wait_terminal(fc, f0["id"])
    out["failed_empty"] = fc.get(f"{_FT}/{f0['id']}/checkpoints", headers=_h()).json()["data"] == []
    # cancelled job: no checkpoint
    cgate = _GateRunner()
    _gated(cgate)
    pc = _client(ft_runner=cgate)
    pfid = _upload(pc)
    p0 = _submit(pc, pfid).json()
    _wait_status(pc, p0["id"], "running")
    pc.post(f"{_FT}/{p0['id']}/cancel", headers=_h())
    _wait_terminal(pc, p0["id"])
    out["cancelled_empty"] = (
        pc.get(f"{_FT}/{p0['id']}/checkpoints", headers=_h()).json()["data"] == []
    )

    # bogus after cursor → empty page; limit bounds
    r = gclient.get(f"{_FT}/{g0['id']}/checkpoints?after=nope", headers=_h())
    out["ckpt_bogus_after_empty"] = r.json()["data"] == []
    r = gclient.get(f"{_FT}/{g0['id']}/checkpoints?limit=0", headers=_h())
    out["ckpt_limit0_422"] = r.status_code == 422
    r = gclient.get(f"{_FT}/{g0['id']}/checkpoints?limit=101", headers=_h())
    out["ckpt_limit101_422"] = r.status_code == 422
    return out


# ---------------------------------------------------------------------------
# Section 7 — cancel
# ---------------------------------------------------------------------------


def _cancel_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()

    # queued/paused cancel lands immediately and the runner never runs
    gate = _GateRunner()
    _gated(gate)
    pc = _client(ft_runner=gate)
    pfid = _upload(pc)
    p0 = _submit(pc, pfid).json()
    p = pc.post(f"{_FT}/{p0['id']}/pause", headers=_h())
    out["pause_queued_paused"] = p.status_code == 200 and p.json()["status"] == "paused"
    c = pc.post(f"{_FT}/{p0['id']}/cancel", headers=_h())
    out["paused_cancel_immediate"] = c.status_code == 200 and c.json()["status"] == "cancelled"
    pfin = _wait_terminal(pc, p0["id"])
    out["paused_cancel_terminal"] = pfin["status"] == "cancelled"
    out["paused_cancel_finished"] = pfin["finished_at"] is not None
    out["paused_cancel_no_model"] = pfin["fine_tuned_model"] is None
    out["paused_cancel_no_artifacts"] = pfin["result_files"] == []
    pev = _events(pc, p0["id"])["data"]
    out["paused_cancel_event"] = any("cancelled" in str(e.get("message")) for e in pev)
    # a cancelled job mints no card — checkpoints stay honest-empty
    out["paused_cancel_no_card"] = (
        pc.get(f"{_FT}/{p0['id']}/checkpoints", headers=_h()).json()["data"] == []
    )

    # running cancel — cooperative: response is still running, the
    # boundary check ends it cancelled
    gate2 = _GateRunner()
    _gated(gate2)
    rc = _client(ft_runner=gate2)
    rfid = _upload(rc)
    r0 = _submit(rc, rfid).json()
    _wait_status(rc, r0["id"], "running")
    c2 = rc.post(f"{_FT}/{r0['id']}/cancel", headers=_h())
    out["running_cancel_response_running"] = c2.status_code == 200 and c2.json()["status"] in (
        "running",
        "cancelled",
    )
    rev = _events(rc, r0["id"])["data"]
    out["running_cancel_event"] = any(
        "cancellation requested" in str(e.get("message")) or "cancelled" in str(e.get("message"))
        for e in rev
    )
    rfin = _wait_terminal(rc, r0["id"])
    out["running_cancel_terminal"] = rfin["status"] == "cancelled"
    out["running_cancel_no_model"] = rfin["fine_tuned_model"] is None
    gate2.release.set()

    # terminal cancels refuse — on all three terminal states
    tf = _upload(client)
    t0 = _submit(client, tf).json()
    _wait_terminal(client, t0["id"])
    r = client.post(f"{_FT}/{t0['id']}/cancel", headers=_h())
    out["terminal_cancel_409"] = r.status_code == 409 and _err_code(r) == "job_terminal"
    r = client.post(f"{_FT}/{t0['id']}/cancel", headers=_h())
    out["repeated_cancel_409"] = r.status_code == 409
    fcl = _client(ft_runner=_fail_runner)
    ff = _upload(fcl)
    f0 = _submit(fcl, ff).json()
    _wait_terminal(fcl, f0["id"])
    r = fcl.post(f"{_FT}/{f0['id']}/cancel", headers=_h())
    out["failed_cancel_409"] = r.status_code == 409 and _err_code(r) == "job_terminal"
    return out


# ---------------------------------------------------------------------------
# Section 8 — pause / resume
# ---------------------------------------------------------------------------


def _pause_resume_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()

    # immediate pause — pre-start ("queued") or at the stage gate
    # ("running"); both paths report ``paused`` on the wire and the
    # recorded event is exactly one of the two honest strings. The
    # queued-path pause is unreachable over HTTP (the worker flips
    # status within microseconds of dispatch — verified 0/40 races);
    # resume restores whichever status pause captured
    gate = _GateRunner()
    _gated(gate)
    qc = _client(ft_runner=gate)
    qfid = _upload(qc)
    q0 = _submit(qc, qfid).json()
    p = qc.post(f"{_FT}/{q0['id']}/pause", headers=_h())
    out["pause_immediate_200"] = p.status_code == 200 and p.json()["status"] == "paused"
    pev = _events(qc, q0["id"])["data"]
    msgs = {str(e.get("message")) for e in pev}
    # the wire records exactly which pause path was taken — pre-start
    # ("queued") or stage-boundary ("running"); an HTTP pause can only
    # reach the running path but both messages are honest
    out["pause_event_taxonomy"] = bool(
        msgs
        & {
            "job paused — will not start until resumed",
            "pause requested — takes effect at the next stage boundary",
        }
    )
    # the worker never completes while paused — give it a beat
    time.sleep(0.15)
    still = qc.get(f"{_FT}/{q0['id']}", headers=_h()).json()
    out["paused_stays"] = still["status"] == "paused"
    # idempotent second pause replays the record and adds no event
    n_ev = len(_events(qc, q0["id"])["data"])
    p2 = qc.post(f"{_FT}/{q0['id']}/pause", headers=_h())
    out["pause_idempotent"] = p2.status_code == 200 and p2.json()["status"] == "paused"
    out["pause_idempotent_no_new_event"] = len(_events(qc, q0["id"])["data"]) == n_ev
    r = qc.post(f"{_FT}/{q0['id']}/resume", headers=_h())
    out["resume_restores_prior"] = r.status_code == 200 and r.json()["status"] in (
        "queued",
        "running",
    )
    gate.release.set()
    qfin = _wait_terminal(qc, q0["id"])
    out["resumed_completes"] = qfin["status"] == "succeeded"
    rev = _events(qc, q0["id"])["data"]
    out["resume_event"] = any("resumed" in str(e.get("message")) for e in rev)

    # running pause → paused at the gate → resume restores running
    gate2 = _GateRunner()
    _gated(gate2)
    rc = _client(ft_runner=gate2)
    rfid = _upload(rc)
    r0 = _submit(rc, rfid).json()
    _wait_status(rc, r0["id"], "running")
    pr = rc.post(f"{_FT}/{r0['id']}/pause", headers=_h())
    out["pause_running_status"] = pr.status_code == 200 and pr.json()["status"] == "paused"
    pev2 = _events(rc, r0["id"])["data"]
    out["pause_running_event"] = any("pause requested" in str(e.get("message")) for e in pev2)
    rr = rc.post(f"{_FT}/{r0['id']}/resume", headers=_h())
    out["resume_running_restores"] = rr.status_code == 200 and rr.json()["status"] == "running"
    gate2.release.set()
    out["running_resumed_completes"] = _wait_terminal(rc, r0["id"])["status"] == "succeeded"

    # a paused job's cancel wins — cancelled, no model, no artifacts
    gate5 = _GateRunner()
    _gated(gate5)
    xc = _client(ft_runner=gate5)
    xfid = _upload(xc)
    x0 = _submit(xc, xfid).json()
    _wait_status(xc, x0["id"], "running")
    xc.post(f"{_FT}/{x0['id']}/pause", headers=_h())
    cx = xc.post(f"{_FT}/{x0['id']}/cancel", headers=_h())
    out["paused_cancel_200"] = cx.status_code == 200 and cx.json()["status"] == "cancelled"
    xfin = _wait_terminal(xc, x0["id"])
    out["paused_cancel_terminal"] = xfin["status"] == "cancelled"
    out["paused_cancel_no_model"] = xfin["fine_tuned_model"] is None

    # refusal set
    fid = _upload(client)
    j = _submit(client, fid).json()
    _wait_terminal(client, j["id"])
    r = client.post(f"{_FT}/{j['id']}/pause", headers=_h())
    out["terminal_pause_409"] = r.status_code == 409 and _err_code(r) == "job_terminal"
    r = client.post(f"{_FT}/{j['id']}/resume", headers=_h())
    out["terminal_resume_409"] = r.status_code == 409 and _err_code(r) == "job_terminal"
    # resume on a live (non-paused) job → job_not_paused
    gate3 = _GateRunner()
    _gated(gate3)
    nc = _client(ft_runner=gate3)
    nfid = _upload(nc)
    n0 = _submit(nc, nfid).json()
    _wait_status(nc, n0["id"], "running")
    r = nc.post(f"{_FT}/{n0['id']}/resume", headers=_h())
    out["unpaused_resume_409"] = r.status_code == 409 and _err_code(r) == "job_not_paused"
    gate3.release.set()
    _wait_terminal(nc, n0["id"])

    # resume is drain-gated — a paused job cannot re-enter under drain
    gate4 = _GateRunner()
    _gated(gate4)
    dc = _client(ft_runner=gate4)
    dfid = _upload(dc)
    d0 = _submit(dc, dfid).json()
    dc.post(f"{_FT}/{d0['id']}/pause", headers=_h())
    dc.post(_DRAIN, headers=_h())
    r = dc.post(f"{_FT}/{d0['id']}/resume", headers=_h())
    out["drain_resume_503"] = r.status_code == 503 and _err_code(r) == "draining"
    r = dc.post(f"{_FT}/ftjob-nope/resume", headers=_h())
    out["drain_resume_beats_404"] = r.status_code == 503 and _err_code(r) == "draining"
    # pause + cancel stay open under drain; the parked job can still end
    r = dc.post(f"{_FT}/{d0['id']}/cancel", headers=_h())
    out["drain_cancel_open"] = r.status_code == 200 and r.json()["status"] == "cancelled"
    return out


# ---------------------------------------------------------------------------
# Section 9 — concurrency + idempotency
# ---------------------------------------------------------------------------


def _concurrency_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # parallel submits mint distinct ids and all execute
    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def _one(suffix: str) -> None:
        r = _submit(client, fid, suffix=suffix)
        with lock:
            results.append(r.json())

    threads = [threading.Thread(target=_one, args=(f"p{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = {r["id"] for r in results}
    out["parallel_distinct_ids"] = len(ids) == 8 and len(results) == 8
    fins = [_wait_terminal(client, i) for i in ids]
    out["parallel_all_succeed"] = all(f["status"] == "succeeded" for f in fins)
    out["parallel_distinct_models"] = len({f["fine_tuned_model"] for f in fins}) == 8

    # same idem key — sequential dedupe, replay returns the live record
    body = {"model": "fx1", "training_file": fid, "suffix": "idem"}
    r1 = client.post(_FT, json=body, headers=_ih("dk-1"))
    r2 = client.post(_FT, json=body, headers=_ih("dk-1"))
    out["idem_same_id"] = r1.json()["id"] == r2.json()["id"]
    fin1 = _wait_terminal(client, r1.json()["id"])
    r3 = client.post(_FT, json=body, headers=_ih("dk-1"))
    out["idem_replay_live_record"] = (
        r3.json()["id"] == fin1["id"] and r3.json()["status"] in _TERMINAL
    )

    # different body under the same key → conflict
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid, "suffix": "other"},
        headers=_ih("dk-1"),
    )
    out["idem_conflict_409"] = r.status_code == 409 and _err_code(r) == "idempotency_conflict"

    # parallel keyed submits — one execution
    calls: list[Any] = []

    def spy_runner(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        calls.append(spec.job_id)
        return _ok_runner(spec, emit=emit, should_cancel=should_cancel, pause_gate=pause_gate)

    kc = _client(ft_runner=spy_runner)
    kfid = _upload(kc)
    kres: list[Any] = []

    def _keyed() -> None:
        kres.append(
            kc.post(
                _FT,
                json={"model": "fx1", "training_file": kfid},
                headers=_ih("par-1"),
            )
        )

    ts = [threading.Thread(target=_keyed) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    kids = {r.json().get("id") for r in kres}
    out["idem_parallel_one_job"] = len(kids) == 1 and len(calls) == 1
    out["idem_parallel_no_errors"] = all(r.status_code == 200 for r in kres)

    # keyed refusal leaves no phantom claim — a retry with a fixed body
    # executes fresh
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": "file-nope"},
        headers=_ih("nk-1"),
    )
    out["keyed_refusal_code"] = r.status_code == 404
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("nk-1"),
    )
    out["keyed_retry_executes"] = r.status_code == 200
    _wait_terminal(client, r.json()["id"])

    # keys are credential-scoped — the same key under a minted
    # credential is a different scope (executes separately)
    minted = _mint(client, _ROOT)
    s1 = client.post(_FT, json=body, headers=_ih("scope-1"))
    s2 = client.post(_FT, json=body, headers=_ih("scope-1", auth=minted["key"]))
    out["idem_credential_scoped"] = s1.json()["id"] != s2.json()["id"]
    _wait_terminal(client, s1.json()["id"])
    _wait_terminal(client, s2.json()["id"])

    # shared training_file — pin the actual policy: files are not
    # exclusive inputs; two jobs over one file both mint and execute
    f1 = _submit(client, fid, suffix="shared1").json()
    f2 = _submit(client, fid, suffix="shared2").json()
    out["shared_file_both_mint"] = f1["id"] != f2["id"]
    out["shared_file_both_succeed"] = all(
        _wait_terminal(client, j["id"])["status"] == "succeeded" for j in (f1, f2)
    )
    return out


# ---------------------------------------------------------------------------
# Section 10 — admission gate ordering
# ---------------------------------------------------------------------------


def _gate_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # unauth precedes everything — even a garbage body gets 401 first
    r = client.post(_FT, json={"bogus": True})
    out["unauth_401"] = r.status_code == 401 and _err_code(r) == "unauthorized"
    r = client.post(
        _FT, json={"model": "fx1", "training_file": "x"}, headers={"X-API-Key": "wrong"}
    )
    out["bad_key_401"] = r.status_code == 401 and _err_type(r) == "authentication_error"

    # scope: read-only key can read but not submit
    ro = _mint(client, _ROOT, scopes=["read"])
    r = client.post(_FT, json={"model": "fx1", "training_file": fid}, headers=_h(ro["key"]))
    out["ro_submit_403"] = r.status_code == 403 and _err_code(r) == "insufficient_scope"
    r = client.get(_FT, headers=_h(ro["key"]))
    out["ro_read_200"] = r.status_code == 200
    r = client.post(f"{_FT}/ftjob-x/cancel", headers=_h(ro["key"]))
    out["ro_cancel_403"] = r.status_code == 403 and _err_code(r) == "insufficient_scope"

    # drain ordering: validation precedes the latch; a valid submit
    # under drain gets the honest 503; a stored idem replay still serves
    keyed = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("dr-1"),
    )
    kid = keyed.json()["id"]
    client.post(_DRAIN, headers=_h())
    r = _submit(client, "file-nope")
    out["drain_validation_precedes"] = r.status_code == 404 and _err_code(r) == "file_not_found"
    r = _submit(client, fid)
    out["drain_submit_503"] = r.status_code == 503 and _err_code(r) == "draining"
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("dr-1"),
    )
    out["drain_replay_serves"] = r.status_code == 200 and r.json()["id"] == kid
    r = client.get(f"{_FT}/{kid}", headers=_h())
    out["drain_read_open"] = r.status_code == 200

    # over_capacity — one inflight slot held by a parked runner
    gate = _GateRunner()
    _gated(gate)
    oc = _client(ft_runner=gate, max_inflight=1)
    ofid = _upload(oc)
    first = _submit(oc, ofid).json()
    _wait_status(oc, first["id"], "running")
    r = _submit(oc, ofid)
    out["over_capacity_503"] = r.status_code == 503 and _err_code(r) == "over_capacity"
    out["over_capacity_retry_after"] = r.headers.get("retry-after") is not None
    gate.release.set()
    _wait_terminal(oc, first["id"])

    # the shipped default runner fails honestly when the real pipeline
    # can't run — the job ends failed with a real error, never a
    # fabricated success
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _RESOURCES, _StubBackend, _temporary_directory  # noqa: PLC0415

    dapp = api_mod.create_app(
        harness=Harness(runner=_fast_runner),
        backend_resolver=lambda *a, **k: _StubBackend(),
        ft_dir=_temporary_directory() / "ft",
    )
    dclient = TestClient(dapp, raise_server_exceptions=False)
    _RESOURCES.get().callback(dapp.state.jobs_executor.shutdown, False, cancel_futures=True)
    dfid = _upload(dclient)
    dj = _submit(dclient, dfid).json()
    dfin = _wait_terminal(dclient, dj["id"], timeout_s=120.0)
    out["default_runner_fails_honest"] = dfin["status"] in ("failed", "succeeded") and (
        dfin["status"] != "failed" or (dfin.get("error") or {}).get("code") == "job_failed"
    )
    return out


# ---------------------------------------------------------------------------
# Section 11 — durability (--state-dir journal)
# ---------------------------------------------------------------------------


def _durability_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.conv_audit import _temporary_directory  # noqa: PLC0415

    state_dir = _temporary_directory() / "state"
    ft_dir = _temporary_directory() / "ft"
    parked_gate = _GateRunner()
    _gated(parked_gate)

    def _disp(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        # the "parked"/"gone" jobs park in the gate (deterministic
        # non-terminal records); everything else completes
        if spec.ft_model_name.split(":")[2] in ("parked", "gone"):
            return parked_gate(spec, emit=emit, should_cancel=should_cancel, pause_gate=pause_gate)
        return _ok_runner(spec, emit=emit, should_cancel=should_cancel, pause_gate=pause_gate)

    c1 = _client(ft_runner=_disp, state_dir=state_dir, ft_dir=ft_dir)
    fid = _upload(c1)

    # j1: terminal succeeded (with a dead callback_url — the failed
    # delivery bookkeeping journals too); j2: parked mid-run then
    # paused; j3: cancelled; keyed submit so the idem record journals
    j1 = _submit(c1, fid, suffix="durable", callback_url="http://127.0.0.1:1/never").json()
    fin1 = _wait_terminal(c1, j1["id"])
    # callback bookkeeping lands after the terminal flip — wait for it
    deadline = time.monotonic() + 15.0
    while fin1.get("callback_status") is None and time.monotonic() < deadline:
        fin1 = _wait_terminal(c1, j1["id"])
        time.sleep(0.1)
    name1 = fin1["fine_tuned_model"]
    j2r = _submit(c1, fid, suffix="parked").json()
    _wait_status(c1, j2r["id"], "running")
    c1.post(f"{_FT}/{j2r['id']}/pause", headers=_h())
    j3 = _submit(c1, fid, suffix="gone").json()
    _wait_status(c1, j3["id"], "running")
    c1.post(f"{_FT}/{j3['id']}/cancel", headers=_h())
    _wait_terminal(c1, j3["id"])
    keyed = c1.post(
        _FT,
        json={"model": "fx1", "training_file": fid, "suffix": "keyed"},
        headers=_ih("persist-1"),
    ).json()
    _wait_terminal(c1, keyed["id"])
    # delete the card — the tombstone must journal
    c1.delete(f"{_MODELS}/{name1}", headers=_h())

    out["journal_file_exists"] = (Path(state_dir) / "ft_jobs.jsonl").exists()

    # restart — a fresh store replays the journal
    c2 = _client(ft_runner=_disp, state_dir=state_dir, ft_dir=ft_dir)
    r1 = c2.get(f"{_FT}/{j1['id']}", headers=_h())
    out["terminal_restored"] = (
        r1.status_code == 200
        and r1.json()["status"] == "succeeded"
        and r1.json()["fine_tuned_model"] == name1
    )
    ev1 = _events(c2, j1["id"])["data"]
    out["events_restored"] = len(ev1) >= 3 and ev1[-1]["message"] == "job succeeded"
    # the deleted card stays deleted (tombstone replayed)
    out["model_delete_not_resurrected"] = (
        c2.get(f"{_MODELS}/{name1}", headers=_h()).status_code == 404
    )
    # a non-terminal job at shutdown fails closed, honestly
    r2 = c2.get(f"{_FT}/{j2r['id']}", headers=_h())
    out["nonterminal_fails_closed"] = (
        r2.json()["status"] == "failed"
        and (r2.json().get("error") or {}).get("code") == "job_failed"
        and "restarted" in str((r2.json().get("error") or {}).get("message", ""))
    )
    r3 = c2.get(f"{_FT}/{j3['id']}", headers=_h())
    out["cancelled_stays_cancelled"] = r3.json()["status"] == "cancelled"
    # idem key survives — replay returns the original record
    rp = c2.post(
        _FT,
        json={"model": "fx1", "training_file": fid, "suffix": "keyed"},
        headers=_ih("persist-1"),
    )
    out["idem_survives_restart"] = rp.status_code == 200 and rp.json()["id"] == keyed["id"]
    # the uploaded file is durable too — a fresh submit post-restart binds it
    post = _submit(c2, fid, suffix="post-restart")
    out["post_restart_submit_binds_file"] = post.status_code == 200
    _wait_terminal(c2, post.json()["id"])

    # the dead-endpoint webhook bookkeeping journals too — the
    # recovered record keeps callback_url + the failed verdict (the
    # secret itself is never journaled, so nothing redelivers)
    out["callback_bookkeeping_journaled"] = (
        r1.json().get("callback_url") == "http://127.0.0.1:1/never"
        and r1.json().get("callback_status") == "failed"
    )

    # eviction journals: with job_max=1 the older job stays gone after restart
    e_state = _temporary_directory() / "estate"
    ec1 = _client(state_dir=e_state, job_max=1)
    efid = _upload(ec1)
    e1 = _submit(ec1, efid, suffix="ev1").json()
    efin1 = _wait_terminal(ec1, e1["id"])
    e2 = _submit(ec1, efid, suffix="ev2").json()
    _wait_terminal(ec1, e2["id"])
    ec2 = _client(state_dir=e_state, job_max=1)
    out["evicted_stays_evicted"] = (
        ec2.get(f"{_FT}/{e1['id']}", headers=_h()).status_code == 404
        and ec2.get(f"{_FT}/{e2['id']}", headers=_h()).status_code == 200
        and ec2.get(f"{_MODELS}/{efin1['fine_tuned_model']}", headers=_h()).status_code == 404
    )
    # cleanup: wake c1's parked worker so teardown doesn't strand it
    c1.post(f"{_FT}/{j2r['id']}/cancel", headers=_h())
    return out


# ---------------------------------------------------------------------------
# Section 12 — metrics honesty
# ---------------------------------------------------------------------------


def _metrics_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    j = _submit(client, fid, hyperparameters={"n_epochs": 3}).json()
    fin = _wait_terminal(client, j["id"])
    # the runner reported 1234 — the record carries it verbatim
    out["tokens_reported_verbatim"] = fin["trained_tokens"] == 1234
    out["hyper_recorded_verbatim"] = fin["hyperparameters"]["n_epochs"] == 3

    # a runner that cannot measure reports null — never a fabricated count
    nc = _client(ft_runner=_none_tokens_runner)
    nfid = _upload(nc)
    n0 = _submit(nc, nfid).json()
    nfin = _wait_terminal(nc, n0["id"])
    out["tokens_unmeasured_null"] = nfin["trained_tokens"] is None

    # artifacts land as real files whose bytes are the runner's bytes
    rf = fin["result_files"]
    blob = client.get(f"{_FILES}/{rf[0]}/content", headers=_h())
    out["artifact_bytes_real"] = blob.status_code == 200 and blob.content == b'{"ok": true}'
    card = client.get(f"{_FILES}/{rf[0]}", headers=_h()).json()
    out["artifact_purpose"] = card.get("purpose") == "fine-tune-result"
    out["artifact_in_list"] = rf[0] in {
        f["id"] for f in client.get(_FILES, headers=_h()).json()["data"]
    }

    # the runner saw the upload's real corpus bytes (captured at spec)
    seen: list[bytes] = []

    def _corpus_reader(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        seen.append(spec.corpus_path.read_bytes())
        return _ok_runner(spec, emit=emit, should_cancel=should_cancel, pause_gate=pause_gate)

    rc = _client(ft_runner=_corpus_reader)
    rfid = _upload(rc)
    r0 = _submit(rc, rfid).json()
    _wait_terminal(rc, r0["id"])
    out["runner_saw_real_corpus"] = seen == [_CORPUS]
    return out


# ---------------------------------------------------------------------------
# Section 13 — envelope uniformity
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)
    ro = _mint(client, _ROOT, scopes=["read"])

    cases: list[tuple[str, Any, int, str, str]] = [
        (
            "env_400",
            _submit(client, fid, model="byok"),
            400,
            "model_not_trainable",
            "invalid_request_error",
        ),
        (
            "env_404_file",
            _submit(client, "file-nope"),
            404,
            "file_not_found",
            "invalid_request_error",
        ),
        (
            "env_404_job",
            client.get(f"{_FT}/ftjob-nope", headers=_h()),
            404,
            "job_not_found",
            "invalid_request_error",
        ),
        (
            "env_422",
            client.post(_FT, json={}, headers=_h()),
            422,
            "validation",
            "invalid_request_error",
        ),
        ("env_401", client.post(_FT, json={}), 401, "unauthorized", "authentication_error"),
        (
            "env_403",
            client.post(_FT, json={"model": "fx1", "training_file": fid}, headers=_h(ro["key"])),
            403,
            "insufficient_scope",
            "permission_error",
        ),
    ]
    for name, r, status, code, etype in cases:
        body = r.json()
        err = body.get("error", {})
        out[name] = (
            r.status_code == status
            and err.get("code") == code
            and err.get("type") == etype
            and isinstance(err.get("message"), str)
            and "param" in err
        )
    # 409 + 503 legs
    j = _submit(client, fid).json()
    _wait_terminal(client, j["id"])
    r = client.post(f"{_FT}/{j['id']}/cancel", headers=_h())
    out["env_409"] = (
        r.status_code == 409
        and _err_code(r) == "job_terminal"
        and _err_type(r) == "invalid_request_error"
    )
    client.post(_DRAIN, headers=_h())
    r = _submit(client, fid)
    out["env_503"] = (
        r.status_code == 503
        and _err_code(r) == "draining"
        and _err_type(r) == "service_unavailable"
    )
    return out


# ---------------------------------------------------------------------------
# Section 14 — metering
# ---------------------------------------------------------------------------


def _metering_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)
    minted = _mint(client, _ROOT)
    raw, kid = minted["key"], minted["id"]
    ro = _mint(client, _ROOT, scopes=["read"])

    def uses() -> int:
        return int(_usage_card(client, _ROOT, kid)["uses"])

    # a submit bills the caller — authenticate precedes the verdict
    u0 = uses()
    client.post(_FT, json={"model": "fx1", "training_file": fid}, headers=_h(raw))
    out["submit_bills_use"] = uses() - u0 == 1
    # a refused submit still bills a use (auth precedes validation)
    u1 = uses()
    client.post(_FT, json={"model": "fx1", "training_file": "file-nope"}, headers=_h(raw))
    out["refused_submit_still_bills"] = uses() - u1 == 1
    # reads bill too — every authenticated call counts
    u2 = uses()
    client.get(_FT, headers=_h(raw))
    out["read_bills_use"] = uses() - u2 == 1
    # scope refusals bill nothing — the auth-layer verdict precedes
    ro0 = int(_usage_card(client, _ROOT, ro["id"])["uses"])
    client.post(_FT, json={"model": "fx1", "training_file": fid}, headers=_h(ro["key"]))
    out["scope_refusal_no_bill"] = int(_usage_card(client, _ROOT, ro["id"])["uses"]) == ro0
    # unauthenticated calls bill nothing
    u3 = uses()
    client.post(_FT, json={"model": "fx1", "training_file": fid})
    out["unauth_no_bill"] = uses() == u3
    # control-plane work never touches token or served ledgers
    card = _usage_card(client, _ROOT, kid)
    out["tokens_never_billed"] = int(card["tokens_used"]) == 0
    out["served_never_called"] = int(card["served"]["calls"]) == 0
    return out


# ---------------------------------------------------------------------------
# Section 15 — terminal webhook
# ---------------------------------------------------------------------------


def _webhook_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.webhooks import (  # noqa: PLC0415
        WEBHOOK_MAX_ATTEMPTS,
        WEBHOOK_SIGNATURE_HEADER,
        WEBHOOK_TIMESTAMP_HEADER,
        verify_webhook,
    )

    sink = _HookSink()
    client = _client()
    fid = _upload(client)

    # signed delivery — headers + HMAC verify against the raw body
    j = client.post(
        _FT,
        json={
            "model": "fx1",
            "training_file": fid,
            "callback_url": f"{sink.url}/ok",
            "callback_secret": "whsec",
        },
        headers=_h(),
    ).json()
    fin = sink.wait_verdict(client, j["id"])
    sink.wait_hits(1)
    path, hdrs, body = sink.hits[0]
    out["webhook_delivered"] = path == "/ok" and fin["callback_status"] == "delivered"
    out["webhook_signed"] = verify_webhook(
        "whsec", hdrs.get(WEBHOOK_TIMESTAMP_HEADER), hdrs.get(WEBHOOK_SIGNATURE_HEADER), body
    )
    out["webhook_body_is_record"] = json.loads(body).get("id") == j["id"]
    out["webhook_attempts_1"] = fin["callback_attempts"] == 1
    # the secret never touches the wire record
    out["webhook_secret_not_leaked"] = "whsec" not in json.dumps(fin)

    # a 4xx endpoint is a definitive rejection — no retries
    j2 = client.post(
        _FT,
        json={
            "model": "fx1",
            "training_file": fid,
            "callback_url": f"{sink.url}/reject",
            "callback_secret": "whsec",
        },
        headers=_h(),
    ).json()
    fin2 = sink.wait_verdict(client, j2["id"])
    n_reject = sum(1 for h in sink.hits if h[0] == "/reject")
    out["webhook_4xx_no_retry"] = (
        fin2["callback_status"] == "failed"
        and fin2["callback_attempts"] == 1
        and n_reject == 1
        and "returned 404" in str(fin2.get("callback_error", ""))
    )

    # a 5xx retries to the cap
    j3 = client.post(
        _FT,
        json={
            "model": "fx1",
            "training_file": fid,
            "callback_url": f"{sink.url}/flaky",
            "callback_secret": "whsec",
        },
        headers=_h(),
    ).json()
    fin3 = sink.wait_verdict(client, j3["id"])
    n_flaky = sum(1 for h in sink.hits if h[0] == "/flaky")
    out["webhook_5xx_retries_cap"] = (
        fin3["callback_status"] == "failed"
        and fin3["callback_attempts"] == WEBHOOK_MAX_ATTEMPTS
        and n_flaky == WEBHOOK_MAX_ATTEMPTS
    )

    # an unsigned request delivers unsigned (no signature headers)
    n0 = len(sink.hits)
    j4 = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid, "callback_url": f"{sink.url}/plain"},
        headers=_h(),
    ).json()
    sink.wait_verdict(client, j4["id"])
    plain_hits = [h for h in sink.hits[n0:] if h[0] == "/plain"]
    out["webhook_unsigned_headers"] = (
        bool(plain_hits) and WEBHOOK_SIGNATURE_HEADER not in plain_hits[0][1]
    )

    # a dead endpoint fails without hanging the worker
    j5 = client.post(
        _FT,
        json={
            "model": "fx1",
            "training_file": fid,
            "callback_url": "http://127.0.0.1:1/never",
        },
        headers=_h(),
    ).json()
    fin5 = sink.wait_verdict(client, j5["id"])
    out["webhook_dead_endpoint_failed"] = (
        fin5["status"] == "succeeded" and fin5["callback_status"] == "failed"
    )

    # cancelled jobs deliver too — the terminal contract is uniform
    gate = _GateRunner()
    _gated(gate)
    cc = _client(ft_runner=gate)
    cfid = _upload(cc)
    jc = cc.post(
        _FT,
        json={
            "model": "fx1",
            "training_file": cfid,
            "callback_url": f"{sink.url}/cancelled",
        },
        headers=_h(),
    ).json()
    cc.post(f"{_FT}/{jc['id']}/pause", headers=_h())
    cc.post(f"{_FT}/{jc['id']}/cancel", headers=_h())
    sink.wait_hits(len(sink.hits) + 1)
    cfin = sink.wait_verdict(cc, jc["id"])
    out["webhook_cancelled_delivers"] = (
        cfin["status"] == "cancelled"
        and cfin["callback_status"] == "delivered"
        and any(h[0] == "/cancelled" for h in sink.hits)
    )
    return out


# ---------------------------------------------------------------------------
# Section 16 — executor admission edge + HarnessClient legs
# ---------------------------------------------------------------------------


def _edge_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client = _client()
    fid = _upload(client)

    # executor gone (shutdown race): the refused submit leaves no ghost
    # record and no phantom idem claim
    app = cast("FastAPI", client.app)
    app.state.jobs_executor.shutdown(wait=True)
    r = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("dead-1"),
    )
    out["dead_executor_503"] = r.status_code == 503 and _err_code(r) == "over_capacity"
    listed = client.get(_FT, headers=_h()).json()["data"]
    out["dead_executor_no_ghost"] = listed == []
    # keyed retry isn't a phantom replay — it's refused on the same gate
    r2 = client.post(
        _FT,
        json={"model": "fx1", "training_file": fid},
        headers=_ih("dead-1"),
    )
    out["dead_executor_no_phantom_claim"] = r2.status_code == 503
    out["dead_executor_still_no_ghost"] = client.get(_FT, headers=_h()).json()["data"] == []
    # a refused submit under an idem key can't wedge a later good submit
    # on a different key either — the list stays clean either way
    return out


def _client_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import (  # noqa: PLC0415
        HarnessAuthError,
        HarnessClient,
        HarnessJobError,
        HarnessTransportError,
    )
    from fx1.serve.eval_lifecycle_audit import _tc_transport  # noqa: PLC0415

    client = _client()
    fid = _upload(client)
    hc = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(client))

    job = hc.create_finetune_job(model="fx1", training_file=fid, suffix="sdk")
    out["client_submit"] = job["id"].startswith("ftjob-") and job["status"] == "queued"
    fin = hc.wait_finetune_job(job["id"], poll_s=0.02, timeout_s=30.0)
    out["client_wait_succeeds"] = fin["status"] == "succeeded" and fin[
        "fine_tuned_model"
    ].startswith("ft:fx1:sdk:")
    got = hc.finetune_job(job["id"])
    out["client_get"] = got["id"] == job["id"]
    listed = hc.finetune_jobs()
    out["client_list"] = any(j["id"] == job["id"] for j in listed["data"])
    ev = hc.finetune_job_events(job["id"])
    out["client_events"] = bool(ev["data"]) and ev["data"][-1]["message"] == "job succeeded"
    ck = hc.finetune_job_checkpoints(job["id"])
    out["client_checkpoints"] = len(ck["data"]) == 1

    # pause/resume legs on a parked job
    gate = _GateRunner()
    _gated(gate)
    pc = _client(ft_runner=gate)
    pfid = _upload(pc)
    hc2 = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(pc))
    pj = hc2.create_finetune_job(model="fx1", training_file=pfid)
    paused = hc2.pause_finetune_job(pj["id"])
    out["client_pause"] = paused["status"] == "paused"
    resumed = hc2.resume_finetune_job(pj["id"])
    out["client_resume"] = resumed["status"] in ("queued", "running")
    gate.release.set()
    hc2.wait_finetune_job(pj["id"], poll_s=0.02, timeout_s=30.0)

    # cancel leg
    gate2 = _GateRunner()
    _gated(gate2)
    cc = _client(ft_runner=gate2)
    cfid = _upload(cc)
    hc3 = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(cc))
    cj = hc3.create_finetune_job(model="fx1", training_file=cfid)
    time.sleep(0.2)
    canc = hc3.cancel_finetune_job(cj["id"])
    out["client_cancel"] = canc["status"] in ("running", "cancelled", "queued", "paused")
    try:
        hc3.wait_finetune_job(cj["id"], poll_s=0.02, timeout_s=30.0)
        out["client_wait_cancelled_raises"] = False
    except HarnessJobError:
        out["client_wait_cancelled_raises"] = True
    gate2.release.set()

    # failed job raises HarnessJobError honestly
    fc = _client(ft_runner=_fail_runner)
    ffid = _upload(fc)
    hc4 = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(fc))
    fj = hc4.create_finetune_job(model="fx1", training_file=ffid)
    try:
        hc4.wait_finetune_job(fj["id"], poll_s=0.02, timeout_s=30.0)
        out["client_wait_failed_raises"] = False
    except HarnessJobError as exc:
        out["client_wait_failed_raises"] = "synthetic runner fault" in str(exc)

    # error map — each wire status lands on its client exception
    try:
        hc.finetune_job("ftjob-nope")
        out["client_404_keyerror"] = False
    except KeyError:
        out["client_404_keyerror"] = True
    try:
        hc.create_finetune_job(model="fx1", training_file="x", hyperparameters={"n_epochs": 0})
        out["client_422_valueerror"] = False
    except ValueError:
        out["client_422_valueerror"] = True
    bad = HarnessClient("http://testserver", api_key="wrong", transport=_tc_transport(client))
    try:
        bad.finetune_jobs()
        out["client_401_autherror"] = False
    except HarnessAuthError:
        out["client_401_autherror"] = True
    # 503 → BackendNotConfiguredError — drain a fresh app and submit
    dc = _client()
    dcid_fid = _upload(dc)
    dc.post(_DRAIN, headers=_h())
    dhc = HarnessClient("http://testserver", api_key=_ROOT, transport=_tc_transport(dc))
    try:
        dhc.create_finetune_job(model="fx1", training_file=dcid_fid)
        out["client_503_backenderr"] = False
    except Exception as exc:
        from fx1.serve.client import BackendNotConfiguredError  # noqa: PLC0415

        out["client_503_backenderr"] = isinstance(exc, BackendNotConfiguredError)
    # other statuses → HarnessTransportError (409 path)
    try:
        hc.cancel_finetune_job(job["id"])
        out["client_409_transport"] = False
    except HarnessTransportError:
        out["client_409_transport"] = True
    # idempotency leg
    k1 = hc.create_finetune_job(model="fx1", training_file=fid, idempotency_key="cli-1")
    k2 = hc.create_finetune_job(model="fx1", training_file=fid, idempotency_key="cli-1")
    out["client_idem_dedup"] = k1["id"] == k2["id"]
    _wait_terminal(client, k1["id"])
    return out


# ---------------------------------------------------------------------------
# Aggregator + sealed bench
# ---------------------------------------------------------------------------


def ft_audit() -> dict[str, Any]:
    """Run the fine-tuning battery; returns literal bools."""
    from fx1.serve.conv_audit import _audit_context  # noqa: PLC0415

    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_submit_validation_probes())
        out.update(_file_binding_probes())
        out.update(_lifecycle_probes())
        out.update(_registry_probes())
        out.update(_events_probes())
        out.update(_checkpoints_probes())
        out.update(_cancel_probes())
        out.update(_pause_resume_probes())
        out.update(_concurrency_probes())
        out.update(_gate_probes())
        out.update(_durability_probes())
        out.update(_metrics_probes())
        out.update(_envelope_probes())
        out.update(_metering_probes())
        out.update(_webhook_probes())
        out.update(_edge_probes())
        out.update(_client_probes())
        return out


def ft_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under ft_audit.v1."""
    r = ft_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "ft_audit",
        "schema": "ft_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": [
                "openai.fine_tuning.submit",
                "openai.fine_tuning.list",
                "openai.fine_tuning.get",
                "openai.fine_tuning.events",
                "openai.fine_tuning.checkpoints",
                "openai.fine_tuning.cancel",
                "openai.fine_tuning.pause",
                "openai.fine_tuning.resume",
                "openai.files (bind/delete)",
                "openai.models (list/retrieve/delete of ft: cards)",
                "openai.chat + responses (ft: resolution)",
                "harness.drain (submit/resume gates)",
                "harness.keys (scope + metering legs)",
                "HarnessClient ft legs + error map",
            ],
            "not_verified": [
                "the shipped default_ft_runner's real Pipeline training (GPU absent — pinned failed-honestly instead)",
                "real-socket HarnessClient ft legs (e2e_audit owns transport)",
                "multi-process state-dir recovery",
            ],
            "not_executed": [
                "an actual fine-tune producing real weights",
            ],
        },
        "interpretation": (
            "Every probe True means: on this checkout POST "
            "/v1/fine_tuning/jobs validates synchronously (required "
            "fields 422, untrainable models 400 model_not_trainable, "
            "bogus/deleted files 404 file_not_found, wrong-purpose and "
            "malformed corpora 400 invalid_training_file, hyperparameter "
            "bounds + suffix/seed typing + extra fields 422), binds only "
            "purpose='fine-tune' uploads, runs queued→running→succeeded "
            "with a frozen submit view and monotone forward stamps, "
            "mints the ft:{base}:{suffix}:{job12} card only when the "
            "runner produced a real checkpoint (resolvable through models "
            "list/retrieve/chat/responses, deletable everywhere, evicted "
            "with its job), streams a bounded oldest-first event feed "
            "that persists post-terminal, lists checkpoints honestly "
            "(empty mid-run, step/metrics never fabricated), cancels and "
            "pauses cooperatively (queued/paused immediately, running at "
            "the stage boundary, terminal 409), dedupes Idempotency-Key "
            "claims per credential scope while refused submits leave no "
            "phantom claim, orders auth→validation→drain→over_capacity "
            "correctly (drain gates submit+resume, reads + cancels stay "
            "open), survives --state-dir restart exactly (terminal "
            "as-was, non-terminal fails closed, deletes and evictions "
            "never resurrect, idem keys persist), reports metrics only "
            "as measured (null trained_tokens is honest), envelopes "
            "every refusal in the OpenAI {error} grammar with the right "
            "type, bills uses at authenticate for every call while "
            "tokens/served stay unmoved, delivers the terminal record "
            "once over a signed webhook (4xx definitive, 5xx capped "
            "retry, secrets never journaled or leaked), and the "
            "HarnessClient legs map every status to its documented "
            "exception. A dead executor admits nothing and ghosts "
            "nothing. SYNTHETIC stub runners only — no research claim."
        ),
    }
    canon = json.dumps(out, sort_keys=True, separators=(",", ":"))
    out["receipt_sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    return out


if __name__ == "__main__":
    print(json.dumps(ft_audit_bench(), indent=1))
