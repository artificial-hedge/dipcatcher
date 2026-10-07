"""cap_audit — the config × surface capability matrix.

Every sibling audit pins one wire under one backend configuration. This
battery pins *capability*: the same harness binary boots under each
backend configuration it can ship in, and every surface is measured for
what it actually does there — wired and answering, honestly refusing
(501 capability gap / 503 unconfigured / 502 upstream failure), or — the
defect this hunt exists to catch — silently degrading (a fabricated
model, an estimated token count, a hung stream, a stateful surface that
pretends it needed a backend).

Configs under test (each gets its own app construction):

- ``stub`` — every link resolves to a full-channel stub (complete,
  tools, stream, embeddings, tokenize). The all-wired pole.
- ``plain`` — every link resolves to a completion-only stub. The
  capability-gap pole: structured surfaces must refuse 501, never fake.
- ``unconf`` — the resolver refuses every link with
  ``BackendNotConfiguredError``: the harness with no model wired.
  Inference must refuse 503 ``backend_unavailable``; harness-side
  stateful surfaces (files, batches, conversations, evals, uploads,
  moderations, jobs, keys, score, drain) must still work.
- ``byok`` — a real ``OpenAICompatBackend`` over a stub upstream: the
  user-supplied-credentials path end to end over a live socket.
- ``byok_bad`` — same real backend; the upstream answers 500 to every
  call. Honest 502 refusal, never a passthrough 500.
- ``byok_notok`` — same real backend; the upstream lacks ``/tokenize``.
  ``count_tokens`` must refuse 501 ``not_implemented`` — never estimate.
- ``local`` — a real ``LocalFx1Backend`` (ship-eligible model card +
  stub engine) — the weights-direct path, including ``ft:`` serving.
- ``hosted`` — a real ``HostedK3Backend`` pointed at the stub upstream.
- ``managed`` — stub backends behind a root key: auth ordering pins
  (401/403 precede drain, drain precedes backend resolution), which
  must not reorder across configs.

The receipt's ``coverage.matrix`` carries the measured per-config ×
surface verdict map (``status[:code]``) — gaps declared, not smoothed.

Sealed ``cap_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.modelcard import EvalDelta, ModelCard
from fx1.serve.backends import (
    BackendNotConfiguredError,
    EmbeddingResult,
    HostedK3Backend,
    LocalFx1Backend,
    OpenAICompatBackend,
    ToolCompletion,
)
from fx1.serve.byok_audit import _StubUpstream
from fx1.serve.conv_audit import _RESOURCES, _audit_context, _temporary_directory
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["cap_audit", "cap_audit_bench"]

_ROOT = "k3y-material"
_KEY = "cap-key"
_BYOK = "byok"
_LOCAL = "local_fx1"
_HOSTED = "hosted_k3"
_MODEL = "fx1"
_Q = "cap-probe"
_EMB_MODEL = "cap-emb"
_UP_MODEL = "cap-upstream-0"
_GHOST = "ft:cap-ghost"
_CMD = "doctor"
_SUITE = "tooluse"
_FNAME = "c.jsonl"
_JSONL = "application/jsonl"
_DONE = "[DONE]"
_SSE_MARK = "cap-"
_ECHO = "cap:"
_BASE_IDS = {"fx1", "hosted_k3", "local_fx1", "byok"}
_TERMINAL = ("succeeded", "failed", "cancelled", "canceled")

# wire codes the refusal classes map to
_C_BE = "backend_unavailable"  # 503 — no backend wired
_C_BF = "backend_failure"  # 502 — wired backend's upstream faulted
_C_NI = "not_implemented"  # 501 — capability gap (chain surfaces)
_C_NS = "not_supported"  # 501 — capability gap (embeddings surface)
_C_AE = "api_error"  # anthropic-typed envelope for any 5xx refusal
_C_MNF = "model_not_found"  # 404 — unknown/ft name
_C_MNT = "model_not_trainable"  # 400 — training refused
_C_DRAIN = "draining"  # 503 — drain latch beats backend resolution
_C_IR = "invalid_request"  # 422 — misplaced override / bad arg

# paths exercised repeatedly
_P_CHAT = "/v1/chat/completions"
_P_MSGS = "/v1/messages"
_P_RESP = "/v1/responses"
_P_COMPLETIONS = "/v1/completions"
_P_TOKENS = "/v1/messages/count_tokens"
_P_EMBED = "/v1/embeddings"
_P_MODELS = "/v1/models"
_P_FILES = "/v1/files"
_P_UPLOADS = "/v1/uploads"
_P_BATCH = "/v1/batches"
_P_ABATCH = "/v1/messages/batches"
_P_VS = "/v1/vector_stores"
_P_CONV = "/v1/conversations"
_P_EVALS = "/v1/evals"
_P_MODS = "/v1/moderations"
_P_FT = "/v1/fine_tuning/jobs"
_P_COMPLETE = "/harness/complete"
_P_STREAM = "/harness/complete/stream"
_P_KEYS = "/harness/keys"
_P_DRAIN = "/harness/drain"
_P_JOBS = "/harness/jobs"
_P_RUNS = "/harness/runs"
_P_HEVALS = "/harness/evals"
_P_USAGE = "/harness/usage"
_P_SCORE = "/harness/score"
_P_GATE = "/harness/gate/check"
_P_BACKENDS = "/harness/backends"
_P_CAPS = "/harness/capabilities"
_P_SELF = "/harness/self"
_P_VERSION = "/harness/version"
_P_COMMANDS = "/harness/commands"
_P_HEALTH = "/health"
_P_READY = "/ready"

_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_BATCH_CORPUS = (
    b'{"custom_id":"cap-1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"fx1","messages":[{"role":"user","content":"hi"}]}}\n'
)
_EVAL_SPEC = {
    "name": "cap-eval",
    "data_source_config": {
        "type": "custom",
        "item_schema": {"suite": _SUITE, "seed": 0, "backend": _BYOK},
    },
}
_TOOL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "add",
        "parameters": {"type": "object", "properties": {}},
    },
}
_PIPELINE = object()  # sentinel — let the app build its own ft runner


# ---------------------------------------------------------------------------
# Stub backends — the two capability poles
# ---------------------------------------------------------------------------


class _PlainStub:
    """Completion-only backend — the capability-gap pole. No tools, no
    stream, no embeddings, no tokenize: every structured channel must
    refuse 501, never estimate or fake a shape."""

    def __init__(self, model: str = "cap-plain-0") -> None:
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
        return f"{_ECHO}{messages[-1]['content']}"

    def close(self) -> None:
        pass


class _CapStub(_PlainStub):
    """Full-channel stub — complete + tools + stream + embeddings +
    tokenize. The all-wired pole: every structured surface answers."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        del sampling, tool_choice, parallel_tool_calls, logprobs, top_logprobs
        name = "calc"
        if tools and isinstance(tools[0].get("function"), dict):
            name = str(tools[0]["function"].get("name") or name)
        return ToolCompletion(
            content=None,
            tool_calls=(
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": '{"x": 1}'},
                },
            ),
            finish_reason="tool_calls",
        )

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del sampling
        yield "cap-"
        yield "stream"
        yield f":{messages[-1]['content']}"

    def embeddings(self, input: Any, *, model: str, **extra: Any) -> Any:  # noqa: A002
        del extra
        count = len(input) if isinstance(input, (list, tuple)) else 1
        return EmbeddingResult(
            data=tuple(
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(count)
            ),
            model=model,
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return sum(len(str(m.get("content", ""))) // 4 + 1 for m in messages)


# ---------------------------------------------------------------------------
# Path-dispatching upstream stub — answers every provider route a real
# OpenAI-compatible engine serves (models, chat, embeddings, tokenize)
# ---------------------------------------------------------------------------


class _CapUpstream(_StubUpstream):
    """``_StubUpstream`` whose reply is chosen by request path, not mode.

    Modes: ``ok`` answers every provider route; ``err500`` refuses all
    with 500 (the broken-upstream config); ``tok_down`` answers all but
    ``/tokenize``, which 404s (the no-tokenize-channel config)."""

    def _respond(self, h: Any) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.mode == "err500":
            self._send(h, 500, json.dumps({"error": "upstream-boom"}).encode())
            return
        path = str(h.path)
        body: dict[str, Any] = {}
        if self.requests:
            raw = self.requests[-1].get("body") or b""
            if raw:
                try:
                    parsed = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    parsed = {}
                if isinstance(parsed, dict):
                    body = parsed
        if path.endswith("/models"):
            self._send(
                h,
                200,
                json.dumps(
                    {
                        "object": "list",
                        "data": [{"id": "cap-engine-0", "object": "model", "created": 1}],
                    }
                ).encode(),
            )
        elif path.endswith("/embeddings"):
            self._send(
                h,
                200,
                json.dumps(
                    {
                        "object": "list",
                        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    }
                ).encode(),
            )
        elif path.endswith("/tokenize"):
            if self.mode == "tok_down":
                self._send(h, 404, json.dumps({"error": "no such route"}).encode())
            else:
                self._send(h, 200, json.dumps({"count": 5}).encode())
        elif path.endswith("/estimate-token-count"):
            self._send(h, 200, json.dumps({"data": {"total_tokens": 5}}).encode())
        elif path.endswith("/chat/completions"):
            if body.get("stream") is True:
                self._sse_start(h)
                for delta in ("cap-", "upstream"):
                    frame = {"choices": [{"index": 0, "delta": {"content": delta}}]}
                    if not self._sse_frame(h, f"data: {json.dumps(frame)}\n\n".encode()):
                        return
                tail = {"choices": [{"index": 0, "delta": {}}], "usage": self.usage}
                if not self._sse_frame(h, f"data: {json.dumps(tail)}\n\n".encode()):
                    return
                self._sse_frame(h, b"data: [DONE]\n\n")
            else:
                self._send(
                    h,
                    200,
                    json.dumps(
                        {
                            "id": "chatcmpl-cap",
                            "object": "chat.completion",
                            "model": self.resp_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": self.answer,
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": self.usage,
                        }
                    ).encode(),
                )
        else:
            self._send(h, 404, json.dumps({"error": "unknown route"}).encode())


# ---------------------------------------------------------------------------
# Resolver factories — one per config
# ---------------------------------------------------------------------------


class _Resolver:
    """``backend_resolver`` spy: per-link factories, records every
    (link, checkpoint_dir) the wire resolved."""

    def __init__(self, factories: dict[str, Any]) -> None:
        self._factories = factories
        self.resolved: list[tuple[str, Any]] = []

    def __call__(self, name: str, *a: Any, **k: Any) -> Any:
        ckpt = k.get("checkpoint_dir") or (a[0] if a else None)
        self.resolved.append((name, ckpt))
        factory = self._factories.get(name)
        if factory is None:
            raise BackendNotConfiguredError(f"cap-audit: {name} is not configured")
        return factory(**k)


def _stub_resolver(cls: Any = _CapStub) -> _Resolver:
    return _Resolver(
        {
            _HOSTED: lambda **k: cls("hosted-cap"),
            _LOCAL: lambda **k: cls("local-cap"),
            _BYOK: lambda **k: cls("byok-cap"),
        }
    )


def _byok_resolver(upstream: _CapUpstream) -> _Resolver:
    def make(**k: Any) -> OpenAICompatBackend:
        kw: dict[str, Any] = {
            "base_url": str(k.get("base_url") or upstream.url),
            "api_key": str(k.get("api_key") or _KEY),
            "model": str(k.get("model") or _UP_MODEL),
        }
        if k.get("timeout_s") is not None:
            kw["timeout_s"] = int(float(k["timeout_s"]))
        return OpenAICompatBackend(**kw)

    return _Resolver({_BYOK: make})


def _local_resolver(upstream: _CapUpstream, ckpt: str) -> _Resolver:
    def make(**k: Any) -> LocalFx1Backend:
        card = str(k.get("checkpoint_dir") or ckpt)
        return LocalFx1Backend(
            card,
            serve_url=upstream.url,
            timeout_s=k.get("timeout_s"),
        )

    return _Resolver({_LOCAL: make})


def _hosted_resolver(upstream: _CapUpstream) -> _Resolver:
    def make(**k: Any) -> HostedK3Backend:
        return HostedK3Backend(
            api_key=_KEY,
            model="cap-k3",
            api_url=f"{upstream.url}/chat/completions",
            timeout_s=float(k.get("timeout_s") or 30.0),
        )

    return _Resolver({_HOSTED: make})


# ---------------------------------------------------------------------------
# App construction + fixture helpers
# ---------------------------------------------------------------------------


def _ok_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Succeeded job minting a real checkpoint dir — card included, so a
    ``local_fx1`` link can genuinely load the produced checkpoint."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    ckpt = spec.work_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    _ship_card(ckpt / "modelcard.json")
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        checkpoint=str(ckpt),
        trained_tokens=7,
    )


def _ship_card(path: Path) -> None:
    """A ship-eligible model card — domain delta up, honesty gate held."""
    card = ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b5942241..f0e1d2c3",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    )
    card.save(path)


def _card_ckpt() -> str:
    """A checkpoint dir carrying a ship-eligible modelcard.json."""
    root = _temporary_directory()
    _ship_card(root / "modelcard.json")
    return str(root)


def _client(
    resolver: Any,
    *,
    api_key: str | None = None,
    ft_runner: Any = _ok_runner,
    **app_kwargs: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — one app construction per config.
    ``ft_runner=_PIPELINE`` leaves the app's default gated pipeline in
    place (the training-requires-a-real-backend pin)."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get("FX1_API_KEY")
    try:
        if api_key is None:
            os.environ.pop("FX1_API_KEY", None)
        else:
            os.environ["FX1_API_KEY"] = api_key
        kwargs: dict[str, Any] = {
            "harness": Harness(runner=fake_runner),
            "backend_resolver": resolver,
            "state_dir": isolated / "state",
            "receipts_dir": receipts,
            "ft_dir": isolated / "fine_tuning",
            "sse_keepalive_s": 0,
        }
        if ft_runner is not _PIPELINE:
            kwargs["ft_runner"] = ft_runner
        kwargs.update(app_kwargs)
        app = api_mod.create_app(**kwargs)
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop("FX1_API_KEY", None)
        else:
            os.environ["FX1_API_KEY"] = saved_key


# ---------------------------------------------------------------------------
# Request + verdict helpers
# ---------------------------------------------------------------------------


def _msgs() -> list[dict[str, str]]:
    return [{"role": "user", "content": _Q}]


def _chat_body(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": _MODEL, "messages": _msgs()}
    body.update(kw)
    return body


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _byok_headers(upstream: _CapUpstream) -> dict[str, str]:
    """The X-Fx1-Byok-* header binding — the credentialed-link path that
    works uniformly on every OpenAI surface."""
    return {
        "X-Fx1-Byok-Base-Url": upstream.url,
        "X-Fx1-Byok-Api-Key": _KEY,
        "X-Fx1-Byok-Model": _UP_MODEL,
    }


def _local_headers(ckpt: str) -> dict[str, str]:
    return {"X-Fx1-Backend": _LOCAL, "X-Fx1-Checkpoint-Dir": ckpt}


def _verdict(resp: Any) -> str:
    """Stable per-cell verdict: ``status[:error.code]`` — the wire's own
    words, so the matrix records what the config actually answered."""
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return str(resp.status_code)
    code = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = str(err.get("code") or err.get("type") or "")
        elif body.get("code"):
            code = str(body["code"])
    return f"{resp.status_code}:{code}" if code else str(resp.status_code)


def _err(resp: Any) -> dict[str, Any]:
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return {}
    err = body.get("error") if isinstance(body, dict) else None
    return dict(err) if isinstance(err, dict) else {}


def _is_openai_envelope(resp: Any) -> bool:
    err = _err(resp)
    return (
        isinstance(err.get("message"), str) and isinstance(err.get("type"), str) and "code" in err
    )


def _is_anthropic_envelope(resp: Any) -> bool:
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return False
    err = body.get("error")
    return (
        isinstance(body, dict)
        and body.get("type") == "error"
        and isinstance(err, dict)
        and isinstance(err.get("type"), str)
        and isinstance(err.get("message"), str)
    )


def _enveloped(resp: Any) -> bool:
    """A refusal in the dialect's error envelope — OpenAI `{error}` on
    ``/v1``, ``{detail, code}`` on ``/harness``, Anthropic
    ``{type:"error"}`` on ``/v1/messages*`` — never a bare 500."""
    if _is_openai_envelope(resp) or _is_anthropic_envelope(resp):
        return True
    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError):
        return False
    return isinstance(body, dict) and isinstance(body.get("detail"), str) and bool(body.get("code"))


class _Probe:
    """One config's measurement accumulator: pins probe bools and records
    the measured matrix cell for every exercised surface."""

    def __init__(self, cfg: str) -> None:
        self.cfg = cfg
        self.out: dict[str, bool] = {}
        self.cells: dict[str, str] = {}

    def pin(self, name: str, measured: bool) -> None:
        self.out[f"{self.cfg}__{name}"] = bool(measured)

    def cell(
        self,
        surface: str,
        resp: Any,
        *,
        status: int,
        code: str | None = None,
        extra: bool = True,
    ) -> str:
        verdict = _verdict(resp)
        self.cells[surface] = verdict
        want = f"{status}:{code}" if code else str(status)
        got_status, _, got_code = verdict.partition(":")
        self.pin(
            f"{surface}__{want}",
            got_status == str(status) and (code is None or got_code == code) and extra,
        )
        return verdict


# ---------------------------------------------------------------------------
# Polling helpers — bounded, deterministic
# ---------------------------------------------------------------------------


def _wait_ft(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(600):
        job: dict[str, Any] = client.get(f"{_P_FT}/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _wait_eval(client: TestClient, eval_id: str) -> dict[str, Any]:
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_HEVALS}/{eval_id}").json()
        if rec["status"] in _TERMINAL:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"eval {eval_id} never reached a terminal state")


def _wait_v1_run(client: TestClient, eval_id: str, run_id: str) -> dict[str, Any]:
    """Poll the OpenAI run wire — terminal is ``completed``/``failed``/
    ``canceled`` (the ``_RUN_STATUS`` dialect map, not harness verbs)."""
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_EVALS}/{eval_id}/runs/{run_id}").json()
        if rec["status"] in ("completed", "failed", "canceled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached a terminal state")


def _wait_batch(client: TestClient, batch_id: str) -> dict[str, Any]:
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_BATCH}/{batch_id}").json()
        if rec["status"] in ("completed", "failed", "cancelled", "expired", "stopped"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"batch {batch_id} never reached a terminal state")


def _wait_abatch(client: TestClient, batch_id: str) -> dict[str, Any]:
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_ABATCH}/{batch_id}").json()
        if rec["processing_status"] != "in_progress":
            return rec
        time.sleep(0.02)
    raise AssertionError(f"abatch {batch_id} never ended")


def _wait_bg(client: TestClient, rid: str) -> dict[str, Any]:
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_RESP}/{rid}").json()
        if rec["status"] in ("completed", "failed", "cancelled", "incomplete"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"response {rid} never reached a terminal state")


def _wait_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(600):
        rec: dict[str, Any] = client.get(f"{_P_JOBS}/{job_id}").json()
        if rec["status"] in ("succeeded", "failed", "cancelled"):
            return rec
        time.sleep(0.02)
    raise AssertionError(f"lab job {job_id} never reached a terminal state")


def _upload(client: TestClient, *, purpose: str = "fine-tune") -> str:
    r = client.post(
        _P_FILES,
        files={"file": (_FNAME, _CORPUS, _JSONL)},
        data={"purpose": purpose},
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _upload_bytes(client: TestClient, body: bytes) -> str:
    r = client.post(
        _P_FILES,
        files={"file": (_FNAME, body, _JSONL)},
        data={"purpose": "batch"},
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _ft_job(client: TestClient, suffix: str = "cap") -> tuple[dict[str, Any], str]:
    fid = _upload(client)
    r = client.post(
        _P_FT,
        json={"model": _LOCAL, "training_file": fid, "suffix": suffix},
    )
    assert r.status_code == 200, r.text
    job = _wait_ft(client, str(r.json()["id"]))
    name = job.get("fine_tuned_model")
    return job, str(name) if isinstance(name, str) else ""


# ---------------------------------------------------------------------------
# Shared surface batteries — run once per config
# ---------------------------------------------------------------------------


def _completion_surfaces(
    p: _Probe,
    client: TestClient,
    *,
    status: int,
    code: str | None,
    acode: str | None = None,
    model: str = _MODEL,
    chain: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    fx1: dict[str, Any] | None = None,
) -> None:
    """chat/responses/messages/completions/hcomplete under one verdict.
    ``code`` is the OpenAI/{detail,code} envelope code; ``acode`` the
    Anthropic ``error.type`` (5xx refusals map to ``api_error`` there)."""
    fx = {"fx1": fx1} if fx1 else {}
    h = headers or {}
    ok = status == 200
    r = client.post(_P_CHAT, json=_chat_body(model=model, **fx), headers=h)
    p.cell("chat", r, status=status, code=code, extra=ok or _is_openai_envelope(r))
    r = client.post(_P_RESP, json={"model": model, "input": _Q, **fx}, headers=h)
    p.cell("responses", r, status=status, code=code, extra=ok or _is_openai_envelope(r))
    r = client.post(_P_MSGS, json={**_chat_body(model=model), "max_tokens": 16, **fx}, headers=h)
    p.cell("messages", r, status=status, code=acode, extra=ok or _is_anthropic_envelope(r))
    if ok:
        p.pin("messages_envelope", r.json().get("type") == "message")
    r = client.post(_P_COMPLETIONS, json={"model": model, "prompt": _Q, **fx}, headers=h)
    p.cell("completions", r, status=status, code=code, extra=ok or _is_openai_envelope(r))
    r = client.post(_P_COMPLETE, json={**(chain or {}), "messages": _msgs()}, headers=h)
    p.cell("hcomplete", r, status=status, code=code, extra=ok or _enveloped(r))


def _stream_surfaces(
    p: _Probe,
    client: TestClient,
    *,
    stream_status: int,
    stream_code: str | None,
    sse_status: int,
    sse_code: str | None = None,
    chain: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    fx1: dict[str, Any] | None = None,
    model: str = _MODEL,
) -> None:
    """``/harness/complete/stream`` (needs a real stream channel) and
    buffered ``stream:true`` chat (needs only ``complete``) — a missing
    channel refuses, a wired one emits SSE frames; never a hang."""
    h = headers or {}
    fx = {"fx1": fx1} if fx1 else {}
    r = client.post(_P_STREAM, json={**(chain or {}), "messages": _msgs()}, headers=h)
    p.cell(
        "stream",
        r,
        status=stream_status,
        code=stream_code,
        extra=(stream_status == 200 and _SSE_MARK in r.text and _DONE in r.text) or _enveloped(r),
    )
    r = client.post(_P_CHAT, json=_chat_body(model=model, stream=True, **fx), headers=h)
    p.cell(
        "chat_sse",
        r,
        status=sse_status,
        code=sse_code,
        extra=(sse_status == 200 and "chat.completion.chunk" in r.text and _DONE in r.text)
        or _is_openai_envelope(r),
    )


def _eval_surfaces(
    p: _Probe,
    client: TestClient,
    *,
    expect: str,
    submit: dict[str, Any],
    run: dict[str, Any],
    vexpect: str,
) -> None:
    """``/harness/evals`` + ``/v1/evals`` — submission always works; the
    terminal record carries the config's truth honestly."""
    r = client.post(_P_HEVALS, json=submit)
    p.cell("eval_submit", r, status=202)
    if r.status_code == 202:
        rec = _wait_eval(client, str(r.json()["eval_id"]))
        p.pin(f"eval_terminal_{expect}", rec["status"] == expect)
        if expect == "failed":
            p.pin("eval_error_recorded", bool(rec.get("error")))
    r = client.post(_P_EVALS, json=_EVAL_SPEC)
    p.cell("v1eval_spec", r, status=201)
    if r.status_code == 201:
        eid = str(r.json()["id"])
        rr = client.post(f"{_P_EVALS}/{eid}/runs", json=run)
        p.pin("v1eval_run_201", rr.status_code == 201)
        if rr.status_code == 201:
            rec = _wait_v1_run(client, eid, str(rr.json()["id"]))
            p.pin(f"v1eval_terminal_{vexpect}", rec["status"] == vexpect)


def _batch_surfaces(p: _Probe, client: TestClient, *, lines_ok: bool) -> None:
    """``/v1/batches`` + ``/v1/messages/batches`` — always reach a terminal
    state; per-line truth lands in ``request_counts`` honestly."""
    fid = _upload_bytes(client, _BATCH_CORPUS)
    r = client.post(
        _P_BATCH,
        json={
            "input_file_id": fid,
            "endpoint": _P_CHAT,
            "completion_window": "24h",
        },
    )
    p.cell("batch_submit", r, status=200)
    if r.status_code == 200:
        rec = _wait_batch(client, str(r.json()["id"]))
        counts = rec.get("request_counts") or {}
        done = int(counts.get("completed") or 0)
        bad = int(counts.get("failed") or 0)
        p.pin(
            "batch_terminal_honest",
            rec["status"] == "completed" and ((done == 1) if lines_ok else (bad == 1)),
        )
    r = client.post(
        _P_ABATCH,
        json={
            "requests": [
                {
                    "custom_id": "cap-1",
                    "params": {"model": _MODEL, "max_tokens": 8, "messages": _msgs()},
                }
            ]
        },
    )
    p.cell("abatch_submit", r, status=200)
    if r.status_code == 200:
        rec = _wait_abatch(client, str(r.json()["id"]))
        counts = rec.get("request_counts") or {}
        ok_n = int(counts.get("succeeded") or 0)
        bad_n = int(counts.get("errored") or 0)
        p.pin(
            "abatch_terminal_honest",
            rec["processing_status"] == "ended" and ((ok_n == 1) if lines_ok else (bad_n == 1)),
        )


def _stateful_surfaces(p: _Probe, client: TestClient) -> None:
    """Harness-side stateful surfaces — they must work with no model
    wired; a refusal here is a design smell this audit pins."""
    r = client.post(
        _P_FILES,
        files={"file": (_FNAME, _CORPUS, _JSONL)},
        data={"purpose": "batch"},
    )
    p.cell("files", r, status=200, extra=r.json().get("id", "").startswith("file-"))
    fid = str(r.json()["id"]) if r.status_code == 200 else ""
    if fid:
        p.pin("files_get_200", client.get(f"{_P_FILES}/{fid}").status_code == 200)
        p.pin(
            "files_content_roundtrip",
            client.get(f"{_P_FILES}/{fid}/content").content == _CORPUS,
        )
    r = client.post(
        _P_UPLOADS,
        json={
            "purpose": "batch",
            "filename": "u.jsonl",
            "bytes": len(_CORPUS),
            "mime_type": _JSONL,
        },
    )
    uid = str(r.json().get("id", ""))
    if r.status_code == 200 and uid:
        part = client.post(f"{_P_UPLOADS}/{uid}/parts", files={"data": ("p", _CORPUS)})
        done = client.post(
            f"{_P_UPLOADS}/{uid}/complete",
            json={"part_ids": [part.json()["id"]]},
        )
        p.cell(
            "uploads",
            done,
            status=200,
            extra=done.json().get("file", {}).get("id", "").startswith("file-"),
        )
    else:
        p.cell("uploads", r, status=200)
    r = client.post(_P_VS, json={})
    vid = str(r.json().get("id", ""))
    p.cell("vs", r, status=200, extra=vid.startswith("vs"))
    if vid and fid:
        p.pin(
            "vs_attach_file",
            client.post(f"{_P_VS}/{vid}/files", json={"file_id": fid}).status_code in (200, 201),
        )
    r = client.post(_P_CONV, json={})
    cid = str(r.json().get("id", ""))
    p.cell("conv", r, status=200, extra=cid.startswith("conv"))
    if cid:
        items = client.post(
            f"{_P_CONV}/{cid}/items",
            json={
                "items": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": _Q}],
                    }
                ]
            },
        )
        p.pin("conv_items_200", items.status_code == 200)
    r = client.post(_P_MODS, json={"input": _Q})
    p.cell("mods", r, status=200, extra="results" in r.json())


def _bg_surface(
    p: _Probe,
    client: TestClient,
    *,
    expect: str,
    fx1: dict[str, Any] | None = None,
    model: str = _MODEL,
) -> None:
    """``background: true`` — submit 200 ``queued``; the terminal record
    carries the config's truth (``completed`` vs ``failed`` with an
    honest ``error.code``), never an HTTP error after submit."""
    body: dict[str, Any] = {"model": model, "input": _Q, "background": True}
    if fx1:
        body["fx1"] = fx1
    r = client.post(_P_RESP, json=body)
    p.pin(
        "bg_submit_200",
        r.status_code == 200 and r.json().get("status") == "queued",
    )
    if r.status_code != 200:
        return
    rec = _wait_bg(client, str(r.json()["id"]))
    p.cells["bg"] = str(rec.get("status"))
    p.pin(f"bg_terminal_{expect}", rec["status"] == expect)
    if expect == "failed":
        err = rec.get("error") or {}
        p.pin("bg_error_honest", err.get("code") == _C_BE)


def _ops_surfaces(p: _Probe, client: TestClient, *, auth: str | None = None) -> None:
    """The /harness/* + /health ops core — backend-independent."""
    h = _h(auth)
    p.cell("ops_health", client.get(_P_HEALTH, headers=h), status=200)
    p.cell("ops_ready", client.get(_P_READY, headers=h), status=200)
    p.cell("ops_version", client.get(_P_VERSION, headers=h), status=200)
    p.cell("ops_caps", client.get(_P_CAPS, headers=h), status=200)
    p.cell("ops_backends", client.get(_P_BACKENDS, headers=h), status=200)
    p.cell("ops_self", client.get(_P_SELF, headers=h), status=200)
    p.cell("ops_usage", client.get(_P_USAGE, headers=h), status=200)
    p.cell("ops_commands", client.get(_P_COMMANDS, headers=h), status=200)
    r = client.post(_P_RUNS, json={"command": _CMD}, headers=h)
    p.cell("ops_runs", r, status=200, extra=r.json().get("exit_code") == 0)
    r = client.post(_P_JOBS, json={"command": _CMD}, headers=h)
    p.cell("ops_jobs_submit", r, status=202)
    if r.status_code == 202:
        job = _wait_job(client, str(r.json()["job_id"]))
        p.pin("ops_jobs_terminal_succeeded", job["status"] == "succeeded")
    r = client.post(_P_SCORE, json={"input": _Q}, headers=h)
    p.cell("ops_score", r, status=200, extra="data" in r.json())
    r = client.post(_P_GATE, json={"text": "proper pinball score 0.9"}, headers=h)
    p.cell("ops_gate", r, status=200, extra="ok" in r.json())


# ---------------------------------------------------------------------------
# Config: stub — every link full-channel
# ---------------------------------------------------------------------------


def _probe_stub() -> _Probe:
    p = _Probe("stub")
    resolver = _stub_resolver()
    client, _api = _client(resolver)
    _completion_surfaces(p, client, status=200, code=None, chain={"backend": _HOSTED})
    r = client.post(_P_CHAT, json=_chat_body())
    p.pin("chat_echo", _ECHO in r.json()["choices"][0]["message"]["content"])
    _stream_surfaces(
        p,
        client,
        stream_status=200,
        stream_code=None,
        sse_status=200,
        chain={"backend": _HOSTED},
    )
    r = client.post(_P_TOKENS, json={"model": _MODEL, "messages": _msgs()})
    p.cell("tokens", r, status=200, extra=int(r.json().get("input_tokens", 0)) > 0)
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q})
    p.cell("embed", r, status=200, extra=len(r.json()["data"]) == 1)
    r = client.post(_P_CHAT, json=_chat_body(tools=[_TOOL]))
    p.cell(
        "tools",
        r,
        status=200,
        extra=bool(r.json()["choices"][0]["message"].get("tool_calls")),
    )
    r = client.get(_P_MODELS)
    p.cell(
        "models",
        r,
        status=200,
        extra={m["id"] for m in r.json()["data"]} >= _BASE_IDS,
    )
    job, ft_name = _ft_job(client)
    p.pin("ft_job_succeeded", job["status"] == "succeeded")
    if ft_name:
        r = client.post(_P_CHAT, json=_chat_body(model=ft_name))
        p.cell("ft_serve", r, status=200, extra=_ECHO in r.text)
        p.pin(
            "ft_resolves_local_link",
            any(name == _LOCAL and ckpt for name, ckpt in resolver.resolved),
        )
        p.pin(
            "models_ft_card",
            ft_name in {m["id"] for m in client.get(_P_MODELS).json()["data"]},
        )
    _eval_surfaces(
        p,
        client,
        expect="succeeded",
        submit={"suite": _SUITE, "backend": _BYOK, "seed": 0},
        run={"model": _BYOK},
        vexpect="completed",
    )
    _batch_surfaces(p, client, lines_ok=True)
    _stateful_surfaces(p, client)
    _bg_surface(p, client, expect="completed")
    _ops_surfaces(p, client)
    return p


# ---------------------------------------------------------------------------
# Config: plain — completion-only stubs (the capability-gap pole)
# ---------------------------------------------------------------------------


def _probe_plain() -> _Probe:
    p = _Probe("plain")
    client, _api = _client(_stub_resolver(_PlainStub))
    _completion_surfaces(p, client, status=200, code=None, chain={"backend": _HOSTED})
    # structured channels refuse honestly — never faked
    _stream_surfaces(
        p,
        client,
        stream_status=501,
        stream_code=_C_NI,
        sse_status=200,
        chain={"backend": _HOSTED},
    )
    r = client.post(_P_TOKENS, json={"model": _MODEL, "messages": _msgs()})
    p.cell("tokens", r, status=501, code=_C_AE, extra=_is_anthropic_envelope(r))
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q})
    p.cell("embed", r, status=501, code=_C_NS, extra=_is_openai_envelope(r))
    r = client.post(_P_CHAT, json=_chat_body(tools=[_TOOL]))
    p.cell("tools", r, status=501, code=_C_NI, extra=_is_openai_envelope(r))
    r = client.get(_P_MODELS)
    p.cell(
        "models",
        r,
        status=200,
        extra={m["id"] for m in r.json()["data"]} >= _BASE_IDS,
    )
    job, ft_name = _ft_job(client, suffix="p")
    p.pin("ft_job_succeeded", job["status"] == "succeeded")
    if ft_name:
        r = client.post(_P_CHAT, json=_chat_body(model=ft_name))
        p.cell("ft_serve", r, status=200)
    _eval_surfaces(
        p,
        client,
        expect="succeeded",
        submit={"suite": _SUITE, "backend": _BYOK, "seed": 0},
        run={"model": _BYOK},
        vexpect="completed",
    )
    _stateful_surfaces(p, client)
    _bg_surface(p, client, expect="completed")
    _ops_surfaces(p, client)
    return p


# ---------------------------------------------------------------------------
# Config: unconf — nothing wired (the refusal pole)
# ---------------------------------------------------------------------------


def _probe_unconf() -> _Probe:
    p = _Probe("unconf")
    resolver = _Resolver({})
    client, _api = _client(resolver)
    # every inference surface refuses 503 backend_unavailable, enveloped
    _completion_surfaces(p, client, status=503, code=_C_BE, acode=_C_AE, chain={"backend": _HOSTED})
    _stream_surfaces(
        p,
        client,
        stream_status=503,
        stream_code=_C_BE,
        sse_status=503,
        sse_code=_C_BE,
        chain={"backend": _HOSTED},
    )
    r = client.post(_P_TOKENS, json={"model": _MODEL, "messages": _msgs()})
    p.cell("tokens", r, status=503, code=_C_AE, extra=_is_anthropic_envelope(r))
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q})
    p.cell("embed", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    r = client.post(_P_CHAT, json=_chat_body(tools=[_TOOL]))
    p.cell("tools", r, status=503, code=_C_BE)
    # the registry stays honest with nothing wired: link ids listed,
    # models never fabricated
    r = client.get(_P_MODELS)
    p.cell(
        "models",
        r,
        status=200,
        extra={m["id"] for m in r.json()["data"]} >= _BASE_IDS,
    )
    p.pin(
        "models_no_fabrication",
        all(m["id"] in _BASE_IDS for m in client.get(_P_MODELS).json()["data"]),
    )
    # stateful surfaces are backend-independent — a refusal here is a
    # design smell; pin them all working
    _stateful_surfaces(p, client)
    # ft submit accepts a trainable spec; the gated pipeline fails
    # honestly at the backend-required stage on the default runner
    client_real, _api2 = _client(resolver, ft_runner=_PIPELINE)
    fid = _upload(client_real)
    r = client_real.post(_P_FT, json={"model": _LOCAL, "training_file": fid})
    p.cell("ft_submit", r, status=200)
    if r.status_code == 200:
        job = _wait_ft(client_real, str(r.json()["id"]))
        p.pin(
            "ft_job_failed_honest",
            job["status"] == "failed" and bool(job.get("error")),
        )
    # a registered ft: name under unconf resolves local_fx1 → 503; an
    # unregistered one → 404 — never a silent base model
    job, ft_name = _ft_job(client, suffix="u")
    if ft_name:
        r = client.post(_P_CHAT, json=_chat_body(model=ft_name))
        p.cell("ft_serve", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    r = client.post(_P_CHAT, json=_chat_body(model=_GHOST))
    p.cell("ft_serve_unknown", r, status=404, code=_C_MNF)
    # evals submit; the worker's backend resolution lands on the record
    _eval_surfaces(
        p,
        client,
        expect="failed",
        submit={"suite": _SUITE, "backend": _BYOK, "seed": 0},
        run={"model": _BYOK},
        vexpect="failed",
    )
    _batch_surfaces(p, client, lines_ok=False)
    _bg_surface(p, client, expect="failed")
    _ops_surfaces(p, client)
    # drain beats backend resolution — a drained harness reports
    # 'draining', not 'backend_unavailable'
    drained, _api3 = _client(resolver)
    r = drained.post(_P_DRAIN, json={"wait_s": 0})
    p.pin("drain_200", r.status_code == 200)
    r = drained.post(_P_CHAT, json=_chat_body())
    p.cell("drain_precedes_backend", r, status=503, code=_C_DRAIN)
    return p


# ---------------------------------------------------------------------------
# Config: byok — real OpenAICompatBackend over a stub upstream
# ---------------------------------------------------------------------------


def _probe_byok() -> _Probe:
    p = _Probe("byok")
    upstream = _CapUpstream("ok")
    resolver = _byok_resolver(upstream)
    client, _api = _client(resolver)
    bh = _byok_headers(upstream)
    creds = {
        "backend": _BYOK,
        "byok": {"base_url": upstream.url, "api_key": _KEY, "model": _UP_MODEL},
    }
    fx1 = {"byok": creds["byok"]}
    # the fx1.byok body binding on chat; the header binding elsewhere
    body = _chat_body(model=_BYOK, fx1=fx1)
    r = client.post(_P_CHAT, json=body)
    p.cell(
        "chat",
        r,
        status=200,
        extra=r.json()["choices"][0]["message"]["content"] == upstream.answer,
    )
    p.pin(
        "upstream_saw_wire",
        any(req["path"].endswith("/chat/completions") for req in upstream.requests),
    )
    p.pin("upstream_saw_auth", f"Bearer {_KEY}" in upstream.auths)
    r = client.post(_P_MSGS, json={**body, "max_tokens": 16})
    p.cell(
        "messages",
        r,
        status=200,
        extra=r.json().get("type") == "message" and r.json()["content"][0]["type"] == "text",
    )
    r = client.post(_P_RESP, json={"model": _BYOK, "input": _Q}, headers=bh)
    p.cell("responses", r, status=200, extra=upstream.answer in r.text)
    r = client.post(_P_COMPLETIONS, json={"model": _BYOK, "prompt": _Q}, headers=bh)
    p.cell("completions", r, status=200, extra=upstream.answer in r.text)
    r = client.post(_P_COMPLETE, json={**creds, "messages": _msgs()})
    p.cell("hcomplete", r, status=200, extra=upstream.answer in r.text)
    r = client.post(_P_STREAM, json={**creds, "messages": _msgs()})
    p.cell(
        "stream",
        r,
        status=200,
        extra=_SSE_MARK in r.text and _DONE in r.text,
    )
    r = client.post(_P_CHAT, json={**body, "stream": True})
    p.cell(
        "chat_sse",
        r,
        status=200,
        extra="chat.completion.chunk" in r.text and _DONE in r.text,
    )
    r = client.post(_P_TOKENS, json=body)
    p.cell("tokens", r, status=200, extra=int(r.json().get("input_tokens", 0)) == 5)
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q}, headers=bh)
    p.cell("embed", r, status=200, extra=len(r.json()["data"]) == 1)
    # OpenAICompatBackend carries a real tools channel — the ``tools``
    # field reaches the wire verbatim
    r = client.post(_P_CHAT, json={**body, "tools": [_TOOL]})
    p.cell("tools", r, status=200, extra=upstream.answer in r.text)
    p.pin("tools_field_on_wire", any("tools" in b for b in upstream.bodies()))
    # registry: link ids listed; the upstream's model name never fabricated
    r = client.get(_P_MODELS)
    ids = {m["id"] for m in r.json()["data"]}
    p.cell("models", r, status=200, extra=ids >= _BASE_IDS)
    p.pin("models_no_upstream_fabrication", upstream.resp_model not in ids)
    # fine-tuning: 'byok' is not a trainable model (400 before any
    # resolution); a trainable spec submits and fails honestly when the
    # gated pipeline cannot reach a checkpoint-capable backend
    r = client.post(_P_FT, json={"model": _BYOK, "training_file": _upload(client)})
    p.cell("ft_byok_model", r, status=400, code=_C_MNT, extra=_is_openai_envelope(r))
    client_real, _api2 = _client(resolver, ft_runner=_PIPELINE)
    fid = _upload(client_real)
    r = client_real.post(_P_FT, json={"model": _LOCAL, "training_file": fid})
    p.cell("ft_submit", r, status=200)
    if r.status_code == 200:
        job = _wait_ft(client_real, str(r.json()["id"]))
        p.pin(
            "ft_job_failed_honest",
            job["status"] == "failed" and bool(job.get("error")),
        )
    # ft: serving — a byok override on a non-byok link refuses 422; a
    # registered ft: name with no checkpoint-capable link wired → 503
    r = client.post(_P_CHAT, json=_chat_body(model=_GHOST, fx1=fx1))
    p.cell("ft_serve_byok", r, status=422, code=_C_IR, extra=_is_openai_envelope(r))
    job, ft_name = _ft_job(client, suffix="b")
    if ft_name:
        r = client.post(_P_CHAT, json=_chat_body(model=ft_name))
        p.cell("ft_serve", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    _eval_surfaces(
        p,
        client,
        expect="succeeded",
        submit={"suite": _SUITE, "backend": _BYOK, "seed": 0, "byok": creds["byok"]},
        run={"model": _BYOK, "byok": creds["byok"]},
        vexpect="completed",
    )
    _stateful_surfaces(p, client)
    _bg_surface(p, client, expect="completed", fx1=fx1, model=_BYOK)
    _ops_surfaces(p, client)
    return p


# ---------------------------------------------------------------------------
# Config: byok_bad — real backend, upstream answers 500 to everything
# ---------------------------------------------------------------------------


def _probe_byok_bad() -> _Probe:
    p = _Probe("byok_bad")
    upstream = _CapUpstream("err500")
    client, _api = _client(_byok_resolver(upstream))
    bh = _byok_headers(upstream)
    body = _chat_body(
        model=_BYOK, fx1={"byok": {"base_url": upstream.url, "api_key": _KEY, "model": _UP_MODEL}}
    )
    r = client.post(_P_CHAT, json=body)
    p.cell("chat", r, status=502, code=_C_BF, extra=_is_openai_envelope(r))
    r = client.post(_P_MSGS, json={**body, "max_tokens": 16})
    p.cell("messages", r, status=502, code=_C_AE, extra=_is_anthropic_envelope(r))
    r = client.post(
        _P_STREAM,
        json={
            "backend": _BYOK,
            "byok": {"base_url": upstream.url, "api_key": _KEY, "model": _UP_MODEL},
            "messages": _msgs(),
        },
    )
    p.cell("stream", r, status=502, code=_C_BF, extra=_enveloped(r))
    r = client.post(_P_TOKENS, json=body)
    p.cell("tokens", r, status=502, code=_C_AE, extra=_is_anthropic_envelope(r))
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q}, headers=bh)
    p.cell("embed", r, status=502, code=_C_BF, extra=_is_openai_envelope(r))
    # five consecutive backend faults open the circuit: further calls
    # fail fast 503 backend_unavailable + Retry-After, never reaching the
    # sick upstream — a real config truth, pinned deliberately
    for _ in range(6):
        client.post(
            _P_COMPLETE, json={"backend": _BYOK, "byok": body["fx1"]["byok"], "messages": _msgs()}
        )
    upstream.requests.clear()
    r = client.post(_P_CHAT, json={**body, "tools": [_TOOL]})
    p.cell("tools", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    p.pin("breaker_no_upstream_call", not upstream.requests)
    p.pin("breaker_retry_after", "Retry-After" in r.headers)
    r = client.post(_P_RESP, json={"model": _BYOK, "input": _Q}, headers=bh)
    p.cell("responses", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    r = client.post(_P_CHAT, json={**body, "stream": True})
    p.cell("chat_sse", r, status=503, code=_C_BE, extra=_is_openai_envelope(r))
    # stateful + ops unaffected by a sick upstream or an open circuit
    r = client.post(_P_MODS, json={"input": _Q})
    p.cell("mods", r, status=200)
    r = client.get(_P_VERSION)
    p.cell("ops_version", r, status=200)
    return p


# ---------------------------------------------------------------------------
# Config: byok_notok — upstream lacks /tokenize → 501, never estimated
# ---------------------------------------------------------------------------


def _probe_byok_notok() -> _Probe:
    p = _Probe("byok_notok")
    upstream = _CapUpstream("tok_down")
    client, _api = _client(_byok_resolver(upstream))
    bh = _byok_headers(upstream)
    body = _chat_body(
        model=_BYOK, fx1={"byok": {"base_url": upstream.url, "api_key": _KEY, "model": _UP_MODEL}}
    )
    r = client.post(_P_CHAT, json=body)
    p.cell("chat", r, status=200, extra=upstream.answer in r.text)
    r = client.post(_P_TOKENS, json=body)
    p.cell("tokens", r, status=501, code=_C_AE, extra=_is_anthropic_envelope(r))
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q}, headers=bh)
    p.cell("embed", r, status=200)
    return p


# ---------------------------------------------------------------------------
# Config: local — real LocalFx1Backend (ship-eligible card + stub engine)
# ---------------------------------------------------------------------------


def _probe_local() -> _Probe:
    p = _Probe("local")
    upstream = _CapUpstream("ok")
    ckpt = _card_ckpt()
    resolver = _local_resolver(upstream, ckpt)
    client, _api = _client(resolver)
    lh = _local_headers(ckpt)
    chain = {"backend": _LOCAL, "checkpoint_dir": ckpt}
    fx1 = {"backend": _LOCAL, "checkpoint_dir": ckpt}
    _completion_surfaces(
        p,
        client,
        status=200,
        code=None,
        model=_MODEL,
        chain=chain,
        headers=lh,
    )
    _stream_surfaces(
        p,
        client,
        stream_status=200,
        stream_code=None,
        sse_status=200,
        chain=chain,
        headers=lh,
    )
    r = client.post(_P_STREAM, json={**chain, "messages": _msgs()})
    p.pin("stream_tokens", _SSE_MARK in r.text and _DONE in r.text)
    r = client.post(_P_TOKENS, json={"model": _MODEL, "messages": _msgs()}, headers=lh)
    p.cell("tokens", r, status=200, extra=int(r.json().get("input_tokens", 0)) == 5)
    # LocalFx1Backend has no embeddings channel — a real capability gap
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q}, headers=lh)
    p.cell("embed", r, status=501, code=_C_NS, extra=_is_openai_envelope(r))
    r = client.post(_P_CHAT, json=_chat_body(tools=[_TOOL]), headers=lh)
    p.cell("tools", r, status=200)
    # a local_fx1 request without a checkpoint anywhere → 422, never a
    # silent default onto another link
    r = client.post(_P_CHAT, json=_chat_body(model=_LOCAL))
    p.cell("no_ckpt", r, status=422, code="validation", extra=_is_openai_envelope(r))
    r = client.get(_P_MODELS)
    p.cell(
        "models",
        r,
        status=200,
        extra={m["id"] for m in r.json()["data"]} >= _BASE_IDS,
    )
    # ft: serving — the runner's checkpoint dir carries a card, so a real
    # LocalFx1Backend loads it end to end
    job, ft_name = _ft_job(client, suffix="l")
    p.pin("ft_job_succeeded", job["status"] == "succeeded")
    if ft_name:
        r = client.post(_P_CHAT, json=_chat_body(model=ft_name))
        p.cell("ft_serve", r, status=200, extra=upstream.answer in r.text)
        p.pin(
            "ft_served_by_local_link",
            any(name == _LOCAL and ck and "ckpt" in str(ck) for name, ck in resolver.resolved),
        )
    _bg_surface(p, client, expect="completed", fx1=fx1)
    _ops_surfaces(p, client)
    return p


# ---------------------------------------------------------------------------
# Config: hosted — real HostedK3Backend pointed at the stub upstream
# ---------------------------------------------------------------------------


def _probe_hosted() -> _Probe:
    p = _Probe("hosted")
    upstream = _CapUpstream("ok")
    client, _api = _client(_hosted_resolver(upstream))
    _completion_surfaces(p, client, status=200, code=None, chain={"backend": _HOSTED})
    p.pin("upstream_saw_auth", f"Bearer {_KEY}" in upstream.auths)
    _stream_surfaces(
        p,
        client,
        stream_status=200,
        stream_code=None,
        sse_status=200,
        chain={"backend": _HOSTED},
    )
    r = client.post(_P_TOKENS, json={"model": _MODEL, "messages": _msgs()})
    p.cell("tokens", r, status=200, extra=int(r.json().get("input_tokens", 0)) == 5)
    r = client.post(_P_EMBED, json={"model": _EMB_MODEL, "input": _Q})
    p.cell("embed", r, status=200, extra=len(r.json()["data"]) == 1)
    r = client.post(_P_CHAT, json=_chat_body(tools=[_TOOL]))
    p.cell("tools", r, status=200)
    r = client.get(_P_MODELS)
    p.cell(
        "models",
        r,
        status=200,
        extra={m["id"] for m in r.json()["data"]} >= _BASE_IDS,
    )
    _bg_surface(p, client, expect="completed")
    _ops_surfaces(p, client)
    return p


# ---------------------------------------------------------------------------
# Config: managed — stub backends behind a root key (auth ordering)
# ---------------------------------------------------------------------------


def _probe_managed() -> _Probe:
    p = _Probe("managed")
    resolver = _stub_resolver()
    client, _api = _client(resolver, api_key=_ROOT)
    root = _h(_ROOT)
    # auth precedes everything: no key → 401 even though backends resolve
    r = client.post(_P_CHAT, json=_chat_body())
    p.cell("auth_no_key", r, status=401)
    r = client.post(_P_CHAT, json=_chat_body(), headers=root)
    p.cell("auth_root", r, status=200)
    # minted keys scope honestly: write-scope mutates; read-scope 403s on
    # mutating routes, still reads
    mint = client.post(_P_KEYS, json={"name": "cap-w", "scopes": ["write"]}, headers=root)
    p.cell("keys_mint", mint, status=201)
    wkey = str(mint.json().get("key", ""))
    r = client.post(_P_CHAT, json=_chat_body(), headers=_h(wkey))
    p.cell("auth_write_scope", r, status=200)
    mint = client.post(_P_KEYS, json={"name": "cap-r", "scopes": ["read"]}, headers=root)
    rkey = str(mint.json().get("key", ""))
    r = client.post(_P_CHAT, json=_chat_body(), headers=_h(rkey))
    p.cell("auth_read_denies_write", r, status=403)
    r = client.get(_P_MODELS, headers=_h(rkey))
    p.cell("auth_read_allows_get", r, status=200)
    r = client.get(_P_KEYS, headers=root)
    p.cell("keys_list", r, status=200)
    r = client.get(_P_SELF, headers=_h(wkey))
    p.cell("write_key_read_route", r, status=403)
    # ordering: auth (401) precedes drain; drain (503 draining) precedes
    # backend resolution — a drained+keyed request never reports
    # backend_unavailable
    drained, _api2 = _client(resolver, api_key=_ROOT)
    r = drained.post(_P_DRAIN, json={"wait_s": 0}, headers=root)
    p.pin("drain_200", r.status_code == 200)
    r = drained.post(_P_CHAT, json=_chat_body())
    p.cell("drain_no_key", r, status=401)
    r = drained.post(_P_CHAT, json=_chat_body(), headers=root)
    p.cell("drain_keyed", r, status=503, code=_C_DRAIN)
    # managed auth on an unconfigured harness: the key passes, then the
    # backend gate refuses — 503 backend_unavailable, never a 401/500 mixup
    unconfed, _api3 = _client(_Resolver({}), api_key=_ROOT)
    r = unconfed.post(_P_CHAT, json=_chat_body(), headers=root)
    p.cell("unconf_keyed", r, status=503, code=_C_BE)
    r = unconfed.post(_P_CHAT, json=_chat_body())
    p.cell("unconf_no_key", r, status=401)
    return p


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _run() -> tuple[dict[str, bool], dict[str, dict[str, str]]]:
    """The full battery: probe map + measured config × surface matrix."""
    out: dict[str, bool] = {}
    matrix: dict[str, dict[str, str]] = {}
    for fn in (
        _probe_stub,
        _probe_plain,
        _probe_unconf,
        _probe_byok,
        _probe_byok_bad,
        _probe_byok_notok,
        _probe_local,
        _probe_hosted,
        _probe_managed,
    ):
        p = fn()
        out.update(p.out)
        matrix[p.cfg] = p.cells
    return out, matrix


def cap_audit() -> dict[str, Any]:
    """Every measured capability probe — flat ``config__surface__verdict``
    name → True map."""
    with _audit_context():
        out, _matrix = _run()
    return out


def cap_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under cap_audit.v1, carrying the
    measured config × surface verdict matrix."""
    with _audit_context():
        results, matrix = _run()
    ok = bool(results) and all(v is True for v in results.values())
    out: dict[str, Any] = {
        "kind": "cap_audit",
        "schema": "cap_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": results, "ok": ok},
        "coverage": {
            "configs": sorted(matrix),
            "matrix": matrix,
            "transport": (
                "in-process buffered TestClient per config; byok/local/"
                "hosted configs resolve real backend classes over a "
                "threaded HTTP stub upstream — the same binary, nine "
                "backend wirings, fully offline"
            ),
            "not_executed": [
                "a real provider upstream (TLS, auth refresh, quotas)",
                "a spawned local engine (FX1_LOCAL_SERVE_CMD lifecycle)",
                "cross-process restart durability of the capability map",
            ],
        },
        "interpretation": (
            "The capability map holds: stub-wired surfaces answer; "
            "completion-only backends refuse structured channels 501 "
            "not_implemented/not_supported; unconfigured inference "
            "refuses 503 backend_unavailable (never 500) while every "
            "harness-side stateful surface keeps working; a broken "
            "upstream surfaces honest 502s; a missing tokenize route "
            "refuses 501 rather than estimate; ft: serving needs a "
            "checkpoint-capable link; auth precedes drain precedes "
            "backend resolution under every config. coverage.matrix "
            "declares each measured verdict."
            if ok
            else "CAP AUDIT DEFECT: "
            + json.dumps({k: v for k, v in results.items() if not v}, sort_keys=True)
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(cap_audit_bench(), indent=2, sort_keys=True))
