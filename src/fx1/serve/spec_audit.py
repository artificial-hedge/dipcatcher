"""spec_audit — vendor-spec conformance battery over the ``/v1`` surfaces.

``contract_audit`` pins our own OpenAPI export; this lane pins the wire
against the *vendors'* published object grammars. The relevant subset of
the OpenAI and Anthropic API specs is vendored below as a frozen,
hand-checked schema map — never fetched, never derived from our own
routes. Every live response the battery mints is validated with
``_require_shape``: required keys present with the right JSON types,
literal discriminators (``object``, ``type``) exact, enums inside the
vendor set, extra keys ignored (both specs allow extension fields — fx1
adds ``X-Fx1-*`` metadata and callback fields on purpose).

Probes (each a literal bool in ``results``):

- *Chat completions* — ``CreateChatCompletionResponse`` shape, choice
  ``message``/``finish_reason`` enum, ``usage`` token triple; SSE frames
  are ``chat.completion.chunk`` objects carrying ``delta``, terminated
  by ``[DONE]``.
- *Responses* — ``object=response`` shape, status enum, typed ``output``
  items, ``input/output/total_tokens`` usage, ``store:false`` never
  retrievable, ``previous_response_id`` chains, and the vendor's
  ``{error:{message,type,param,code}}`` envelope on 4xx.
- *Models* — ``object=list`` of ``object=model`` cards; unknown ids 404
  in the same envelope.
- *Legacy completions* — ``object=text_completion`` with ``choices[].text``
  on both the sync body and the SSE stream.
- *Files/batches/fine-tuning* — ``file`` list items, ``batch`` status +
  ``request_counts`` triple, ``fine_tuning.job`` status enum +
  ``fine_tuned_model``.
- *Anthropic* — ``/v1/messages`` mints ``type=message`` with typed
  ``content[]`` blocks, vendor ``stop_reason``, ``input/output_tokens``
  usage; SSE carries the vendor event grammar; errors are the Anthropic
  ``{type:"error",error:{type,message}}`` envelope — never the OpenAI
  one; ``count_tokens`` returns ``{input_tokens:int}``; ``/v1/models``
  under ``anthropic-version`` speaks Anthropic cursor paging
  (``data``/``first_id``/``last_id``/``has_more``); ``/v1/messages/batches``
  mints ``type=message_batch`` with the five-bucket ``request_counts``.
- *Negative shape* — each surface must not leak the other's grammar:
  OpenAI bodies never carry ``type:error``, Anthropic bodies never carry
  ``object``/``choices``, and usage token keys never cross-vendor.
- *Header contracts* — rpm 429s carry ``Retry-After`` + ``X-RateLimit-*``;
  quota ``429 quota_exceeded`` carries none; 401/403 land in each
  surface's own error envelope.

Honesty: synthetic stub backends only — this measures wire shape, never
model quality; the sealed receipt is ``data_label: SYNTHETIC``.

Composition: mirrors ``api_audit`` scaffolding (TestClient over
``create_app`` with injected ``backend_resolver``); reads nothing outside
the live app and this module.

References: OpenAI API spec (chat.completion, response, model, file,
batch, fine_tuning.job object grammars + error envelope); Anthropic API
spec 2023-06-01 (message, message_batch, model list, error envelope +
SSE event grammar). Both vendored, not fetched.

Sealed ``spec_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["spec_audit", "spec_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT_KEY = "k3y-material"
_CHAT_PATH = "/v1/chat/completions"
_RESP_PATH = "/v1/responses"
_MODELS_PATH = "/v1/models"
_MSGS_PATH = "/v1/messages"
_FILES_PATH = "/v1/files"
_ANTHROPIC_H = {"anthropic-version": "2023-06-01"}

# --- vendored vendor-spec subset -------------------------------------------
# Mini schema grammar, checked by ``_require_shape``:
#   dict            → object: every non-``?``-suffixed key required; extras OK
#   "key?"          → optional key: validated only when present
#   (spec, ...)     → union: value must satisfy one alternative
#   [spec]          → list: every element must satisfy spec[0]
#   frozenset({…})  → enum: value must be a member
#   type            → strict JSON scalar match (bool/int/str exact)
#   anything else   → literal equality

_OAI_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_OAI_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "rate_limit_error",
        "server_error",
        "service_unavailable_error",
        "overloaded_error",
        "insufficient_quota",
    }
)
_OAI_RESPONSE_STATUS = frozenset(
    {"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"}
)
_OAI_BATCH_STATUS = frozenset(
    {
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    }
)
_OAI_FT_STATUS = frozenset(
    {"validating_files", "queued", "running", "succeeded", "failed", "cancelled"}
)
_ANTH_STOP_REASONS = frozenset(
    {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"}
)
_ANTH_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",
        "authentication_error",
        "billing_error",
        "permission_error",
        "not_found_error",
        "rate_limit_error",
        "timeout_error",
        "api_error",
        "overloaded_error",
    }
)
_ANTH_EVENT_TYPES = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
        "error",
    }
)
_ANTH_BATCH_STATUS = frozenset({"in_progress", "canceling", "ended"})

_SCHEMA_OAI_ERROR: dict[str, Any] = {
    "error": {
        "message": str,
        "type": _OAI_ERROR_TYPES,
        "param": (str, type(None)),
        "code": (str, type(None)),
    }
}
_SCHEMA_OAI_USAGE: dict[str, Any] = {
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int,
}
_SCHEMA_CHAT_COMPLETION: dict[str, Any] = {
    "id": str,
    "object": "chat.completion",
    "created": int,
    "model": str,
    "choices": [
        {
            "index": int,
            "message": {"role": str, "content": (str, type(None))},
            "finish_reason": (_OAI_FINISH_REASONS, type(None)),
        }
    ],
    "usage": (_SCHEMA_OAI_USAGE, type(None)),
}
_SCHEMA_CHAT_CHUNK: dict[str, Any] = {
    "id": str,
    "object": "chat.completion.chunk",
    "created": int,
    "model": str,
    "choices": [
        {
            "index": int,
            "delta": dict,
            "finish_reason": (_OAI_FINISH_REASONS, type(None)),
        }
    ],
}
_SCHEMA_RESP_USAGE: dict[str, Any] = {
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
}
_SCHEMA_RESPONSE: dict[str, Any] = {
    "id": str,
    "object": "response",
    "created_at": int,
    "status": _OAI_RESPONSE_STATUS,
    "model": str,
    "output": [{"type": str, "id": str}],
    "usage": (_SCHEMA_RESP_USAGE, type(None)),
    "store": (bool, type(None)),
    "previous_response_id": (str, type(None)),
    "error?": (dict, type(None)),
    "incomplete_details?": (dict, type(None)),
}
_SCHEMA_OAI_MODEL: dict[str, Any] = {
    "id": str,
    "object": "model",
    "created": int,
    "owned_by": str,
}
_SCHEMA_OAI_MODEL_LIST: dict[str, Any] = {"object": "list", "data": [_SCHEMA_OAI_MODEL]}
_SCHEMA_TEXT_COMPLETION: dict[str, Any] = {
    "id": str,
    "object": "text_completion",
    "created": int,
    "model": str,
    "choices": [
        {
            "index": int,
            "text": str,
            "finish_reason": (_OAI_FINISH_REASONS, type(None)),
        }
    ],
    # optional in the vendor spec — SSE chunks omit it entirely
    "usage?": (_SCHEMA_OAI_USAGE, type(None)),
}
_SCHEMA_OAI_FILE: dict[str, Any] = {
    "id": str,
    "object": "file",
    "bytes": int,
    "created_at": int,
    "filename": str,
    "purpose": str,
}
_SCHEMA_OAI_FILE_LIST: dict[str, Any] = {"object": "list", "data": [_SCHEMA_OAI_FILE]}
_SCHEMA_OAI_BATCH: dict[str, Any] = {
    "id": str,
    "object": "batch",
    "endpoint": str,
    "input_file_id": str,
    "completion_window": str,
    "status": _OAI_BATCH_STATUS,
    "created_at": int,
    "request_counts": {"total": int, "completed": int, "failed": int},
}
_SCHEMA_FT_JOB: dict[str, Any] = {
    "id": str,
    "object": "fine_tuning.job",
    "model": str,
    "created_at": int,
    "status": _OAI_FT_STATUS,
    "fine_tuned_model": (str, type(None)),
    "training_file": str,
    "result_files": [str],
    "hyperparameters": dict,
}
_SCHEMA_ANTH_ERROR: dict[str, Any] = {
    "type": "error",
    "error": {"type": _ANTH_ERROR_TYPES, "message": str},
}
_SCHEMA_ANTH_USAGE: dict[str, Any] = {"input_tokens": int, "output_tokens": int}
_SCHEMA_ANTH_MESSAGE: dict[str, Any] = {
    "id": str,
    "type": "message",
    "role": "assistant",
    "model": str,
    "content": [{"type": str}],
    "stop_reason": (_ANTH_STOP_REASONS, type(None)),
    "usage": _SCHEMA_ANTH_USAGE,
}
_SCHEMA_ANTH_MODEL: dict[str, Any] = {
    "type": "model",
    "id": str,
    "display_name": str,
    "created_at": str,
}
_SCHEMA_ANTH_MODEL_LIST: dict[str, Any] = {
    "data": [_SCHEMA_ANTH_MODEL],
    "first_id": (str, type(None)),
    "last_id": (str, type(None)),
    "has_more": bool,
}
_SCHEMA_ANTH_BATCH: dict[str, Any] = {
    "id": str,
    "type": "message_batch",
    "processing_status": _ANTH_BATCH_STATUS,
    "request_counts": {
        "processing": int,
        "succeeded": int,
        "errored": int,
        "canceled": int,
        "expired": int,
    },
    "created_at": str,
    "expires_at": str,
    "ended_at": (str, type(None)),
    "results_url": (str, type(None)),
}
_SCHEMA_ANTH_COUNT: dict[str, Any] = {"input_tokens": int}

# Shape violations of the most recent spec_audit() run — for the receipt's
# defect report. Module state mirrors how the probe dicts accumulate.
_SHAPE_ERRORS: list[str] = []


def _require_shape(obj: Any, schema: dict[str, Any], path: str = "$") -> str | None:
    """First spec-subset violation under ``obj``, or None when it conforms.

    Strict on required keys and JSON types, permissive on extra keys —
    vendor specs allow extension fields, so absence is a defect but
    presence of extras is not.
    """
    if not isinstance(obj, dict):
        return f"{path}: expected object, got {type(obj).__name__}"
    for raw_key, spec in schema.items():
        optional = raw_key.endswith("?")
        key = raw_key[:-1] if optional else raw_key
        if key not in obj:
            if not optional:
                return f"{path}.{key}: missing required key"
            continue
        err = _check(obj[key], spec, f"{path}.{key}")
        if err is not None:
            return err
    return None


def _check(val: Any, spec: Any, path: str) -> str | None:
    if isinstance(spec, dict):
        return _require_shape(val, spec, path)
    if isinstance(spec, frozenset):
        return None if val in spec else f"{path}: {val!r} outside vendor enum"
    if isinstance(spec, tuple):
        return (
            None
            if any(_check(val, s, path) is None for s in spec)
            else f"{path}: {val!r} matches no allowed alternative"
        )
    if isinstance(spec, list):
        if not isinstance(val, list):
            return f"{path}: expected list, got {type(val).__name__}"
        for i, item in enumerate(val):
            err = _check(item, spec[0], f"{path}[{i}]")
            if err is not None:
                return err
        return None
    if isinstance(spec, type):
        # bool is an int subclass — JSON scalar checks must be exact
        ok = type(val) is spec if spec in (str, int, bool) else isinstance(val, spec)
        return None if ok else f"{path}: expected {spec.__name__}, got {type(val).__name__}"
    return None if val == spec else f"{path}: expected {spec!r}, got {val!r}"


def _shape(obj: Any, schema: dict[str, Any], path: str = "$") -> bool:
    err = _require_shape(obj, schema, path)
    if err is None:
        return True
    _SHAPE_ERRORS.append(err)
    return False


def _client(api_key: str | None = None, **create_kw: Any) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction."""
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod

    saved = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(**create_kw)
        return _TC(app, raise_server_exceptions=False), api_mod
    finally:
        if saved is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved


class _StubBackend:
    """Deterministic echo backend — no usage channel."""

    def __init__(self) -> None:
        self._model = "fake-0"

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return f"clean:{messages[-1]['content']}"


class _MeterBackend(_StubBackend):
    """Echo backend with a usage channel — meters provider-reported tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.last_usage: dict[str, int] | None = None

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        return super().complete(messages, sampling=sampling)


class _CountBackend(_StubBackend):
    """Echo backend with the Anthropic tokenize channel."""

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return 42


def _sse_data(resp_text: str) -> list[str]:
    """Raw ``data:`` payloads of an SSE body, ``[DONE]`` sentinel included."""
    return [line[len("data: ") :] for line in resp_text.splitlines() if line.startswith("data: ")]


def _probe_chat(out: dict[str, Any]) -> None:
    """OpenAI chat-completions grammar — sync body + SSE chunk stream."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())
    metered, _ = _client(backend_resolver=lambda *a, **k: _MeterBackend())
    body = {"model": "fx1", "messages": [{"role": "user", "content": "ping"}]}

    r = client.post(_CHAT_PATH, json=body)
    out["chat_completion_schema"] = r.status_code == 200 and _shape(
        r.json(), _SCHEMA_CHAT_COMPLETION
    )
    ru = metered.post(_CHAT_PATH, json=body)
    out["chat_usage_schema"] = ru.status_code == 200 and _shape(
        ru.json().get("usage"), _SCHEMA_OAI_USAGE, "$.usage"
    )

    rs = client.post(_CHAT_PATH, json={**body, "stream": True})
    datas = _sse_data(rs.text)
    chunks = [json.loads(d) for d in datas if d != "[DONE]"]
    out["chat_sse_content_type"] = rs.status_code == 200 and rs.headers.get(
        "content-type", ""
    ).startswith("text/event-stream")
    out["chat_sse_chunk_schema"] = bool(chunks) and all(
        _shape(c, _SCHEMA_CHAT_CHUNK, "$.chunk") for c in chunks
    )
    deltas = [c["choices"][0]["delta"] for c in chunks]
    out["chat_sse_delta_grammar"] = (
        bool(deltas)
        and deltas[0].get("role") == "assistant"
        and all(isinstance(d.get("content"), str) for d in deltas if "content" in d)
        and chunks[-1]["choices"][0]["finish_reason"] in _OAI_FINISH_REASONS
    )
    out["chat_sse_done_terminal"] = datas[-1] == "[DONE]"


def _probe_responses(out: dict[str, Any]) -> None:
    """OpenAI Responses grammar — status enum, typed output, store gate."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())
    metered, _ = _client(backend_resolver=lambda *a, **k: _MeterBackend())

    r = client.post(_RESP_PATH, json={"model": "fx1", "input": "hello"})
    rj = r.json()
    out["responses_schema"] = r.status_code == 200 and _shape(rj, _SCHEMA_RESPONSE)
    out["responses_output_items_typed"] = (
        bool(rj.get("output"))
        and all(i.get("type") and i.get("id") for i in rj["output"])
        and (rj["output"][0].get("content") or [{}])[0].get("type") == "output_text"
    )
    ru = metered.post(_RESP_PATH, json={"model": "fx1", "input": "x"})
    out["responses_usage_schema"] = _shape(ru.json().get("usage"), _SCHEMA_RESP_USAGE, "$.usage")
    rsf = client.post(_RESP_PATH, json={"model": "fx1", "input": "x", "store": False})
    rsf_get = client.get(f"{_RESP_PATH}/{rsf.json()['id']}")
    out["responses_store_false_404"] = (
        rsf.status_code == 200
        and rsf_get.status_code == 404
        and _shape(rsf_get.json(), _SCHEMA_OAI_ERROR)
    )
    parent = client.post(_RESP_PATH, json={"model": "fx1", "input": "a"})
    child = client.post(
        _RESP_PATH,
        json={
            "model": "fx1",
            "input": "b",
            "previous_response_id": parent.json()["id"],
        },
    )
    out["responses_prev_chain"] = (
        parent.status_code == 200
        and child.status_code == 200
        and child.json().get("previous_response_id") == parent.json()["id"]
        and _shape(child.json(), _SCHEMA_RESPONSE)
    )
    e422 = client.post(_RESP_PATH, json={"model": "fx1"})
    e404 = client.get(f"{_RESP_PATH}/resp_nope")
    out["responses_validation_envelope"] = e422.status_code in (400, 422) and _shape(
        e422.json(), _SCHEMA_OAI_ERROR
    )
    out["responses_404_envelope"] = e404.status_code == 404 and _shape(
        e404.json(), _SCHEMA_OAI_ERROR
    )


def _probe_models(out: dict[str, Any]) -> None:
    """OpenAI models list + retrieve + not-found envelope."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())

    ml = client.get(_MODELS_PATH)
    out["models_list_schema"] = ml.status_code == 200 and _shape(ml.json(), _SCHEMA_OAI_MODEL_LIST)
    mg = client.get(f"{_MODELS_PATH}/fx1")
    out["models_get_schema"] = mg.status_code == 200 and _shape(mg.json(), _SCHEMA_OAI_MODEL)
    m404 = client.get(f"{_MODELS_PATH}/not-a-model")
    out["models_404_envelope"] = m404.status_code == 404 and _shape(m404.json(), _SCHEMA_OAI_ERROR)


def _probe_legacy(out: dict[str, Any]) -> None:
    """Legacy text-completions grammar — sync + SSE."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())

    r = client.post("/v1/completions", json={"model": "fx1", "prompt": "hello"})
    out["legacy_completion_schema"] = r.status_code == 200 and _shape(
        r.json(), _SCHEMA_TEXT_COMPLETION
    )
    rs = client.post("/v1/completions", json={"model": "fx1", "prompt": "hi", "stream": True})
    datas = _sse_data(rs.text)
    chunks = [json.loads(d) for d in datas if d != "[DONE]"]
    out["legacy_sse_chunk_schema"] = bool(chunks) and all(
        _shape(c, _SCHEMA_TEXT_COMPLETION, "$.chunk") for c in chunks
    )
    out["legacy_sse_done_terminal"] = datas[-1] == "[DONE]" and all(
        c["choices"][0]["finish_reason"] in _OAI_FINISH_REASONS
        or c["choices"][0]["finish_reason"] is None
        for c in chunks
    )


def _batch_poll(client: TestClient, batch_id: str) -> dict[str, Any]:
    b: dict[str, Any] = {}
    for _ in range(500):
        b = dict(client.get(f"/v1/batches/{batch_id}").json())
        if b.get("status") in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(0.01)
    return b


def _probe_files_batches(out: dict[str, Any]) -> None:
    """OpenAI files + batches object grammar."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())

    line = {
        "custom_id": "a",
        "method": "POST",
        "url": _CHAT_PATH,
        "body": {"model": "fx1", "messages": [{"role": "user", "content": "x"}]},
    }
    payload = (json.dumps(line) + "\n").encode()
    up = client.post(
        _FILES_PATH,
        files={"file": ("in.jsonl", payload, "application/jsonl")},
        data={"purpose": "batch"},
    )
    out["file_object_schema"] = up.status_code == 200 and _shape(up.json(), _SCHEMA_OAI_FILE)
    out["files_list_schema"] = _shape(client.get(_FILES_PATH).json(), _SCHEMA_OAI_FILE_LIST)
    bc = client.post(
        "/v1/batches",
        json={
            "input_file_id": up.json()["id"],
            "endpoint": _CHAT_PATH,
            "completion_window": "24h",
        },
    )
    out["batch_create_schema"] = bc.status_code == 200 and _shape(bc.json(), _SCHEMA_OAI_BATCH)
    term = _batch_poll(client, bc.json()["id"])
    out["batch_terminal_schema"] = term.get("status") == "completed" and _shape(
        term, _SCHEMA_OAI_BATCH
    )
    b404 = client.get("/v1/batches/batch_nope")
    out["batch_404_envelope"] = b404.status_code == 404 and _shape(b404.json(), _SCHEMA_OAI_ERROR)


def _probe_finetune(out: dict[str, Any]) -> None:
    """OpenAI fine_tuning.job object grammar over the stub runner."""
    from fx1.serve.finetune import FTJobOutcome

    def _runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        emit("info", "stub train step", {"epoch": 1})
        art = spec.work_dir / "train_receipt.json"
        art.write_text("{}")
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts={"train_receipt": art},
            trained_tokens=None,
            checkpoint=str(spec.work_dir / "ckpt"),
        )

    client, _ = _client(ft_runner=_runner)
    up = client.post(
        _FILES_PATH,
        files={
            "file": (
                "c.jsonl",
                b'{"messages":[{"role":"user","content":"q"},'
                b'{"role":"assistant","content":"a"}]}\n',
            )
        },
        data={"purpose": "fine-tune"},
    )
    sub = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": up.json()["id"]},
    )
    out["ft_job_create_schema"] = sub.status_code == 200 and _shape(sub.json(), _SCHEMA_FT_JOB)
    jid = sub.json()["id"]
    term: dict[str, Any] = {}
    for _ in range(500):
        term = dict(client.get(f"/v1/fine_tuning/jobs/{jid}").json())
        if term.get("status") in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_job_terminal_schema"] = (
        term.get("status") == "succeeded"
        and isinstance(term.get("fine_tuned_model"), str)
        and _shape(term, _SCHEMA_FT_JOB)
    )
    j404 = client.get("/v1/fine_tuning/jobs/ftjob-nope")
    out["ft_job_404_envelope"] = j404.status_code == 404 and _shape(j404.json(), _SCHEMA_OAI_ERROR)


def _anthropic_frames(resp_text: str) -> list[dict[str, Any]]:
    """Parse Anthropic SSE frames → [{event, data}] pairs."""
    frames: list[dict[str, Any]] = []
    for block in resp_text.split("\n\n"):
        if not block.strip():
            continue
        ev_name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                ev_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data:
            frames.append({"event": ev_name, "data": json.loads(data)})
    return frames


def _probe_anthropic(out: dict[str, Any]) -> None:
    """Anthropic message, stream, count_tokens, models, batches grammar."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())
    counter, _ = _client(backend_resolver=lambda *a, **k: _CountBackend())

    am = client.post(
        _MSGS_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    amj = am.json()
    out["anthropic_message_schema"] = am.status_code == 200 and _shape(amj, _SCHEMA_ANTH_MESSAGE)
    blocks = amj.get("content", [])
    out["anthropic_content_blocks_typed"] = bool(blocks) and all(
        isinstance(b, dict)
        and isinstance(b.get("type"), str)
        and (b.get("type") != "text" or isinstance(b.get("text"), str))
        for b in blocks
    )

    ams = client.post(
        _MSGS_PATH,
        json={
            "model": "fx1",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "s"}],
            "stream": True,
        },
    )
    frames = _anthropic_frames(ams.text)
    seq = [f["event"] for f in frames]
    out["anthropic_sse_events"] = (
        ams.status_code == 200
        and ams.headers.get("content-type", "").startswith("text/event-stream")
        and bool(frames)
        and all(f["event"] in _ANTH_EVENT_TYPES for f in frames)
        and seq[0] == "message_start"
        and seq[-1] == "message_stop"
    )
    out["anthropic_sse_message_schema"] = _shape(
        frames[0]["data"].get("message"), _SCHEMA_ANTH_MESSAGE, "$.message"
    )

    ae = client.post(_MSGS_PATH, json={"model": "fx1", "max_tokens": 64})
    out["anthropic_error_envelope"] = ae.status_code in (400, 422) and _shape(
        ae.json(), _SCHEMA_ANTH_ERROR
    )
    a404 = client.get(f"{_MODELS_PATH}/nope", headers=_ANTHROPIC_H)
    out["anthropic_404_envelope"] = a404.status_code == 404 and _shape(
        a404.json(), _SCHEMA_ANTH_ERROR
    )

    ct = counter.post(
        f"{_MSGS_PATH}/count_tokens",
        json={"model": "fx1", "messages": [{"role": "user", "content": "p"}]},
    )
    out["anthropic_count_tokens_schema"] = ct.status_code == 200 and _shape(
        ct.json(), _SCHEMA_ANTH_COUNT
    )
    ct501 = client.post(
        f"{_MSGS_PATH}/count_tokens",
        json={"model": "fx1", "messages": [{"role": "user", "content": "p"}]},
    )
    out["anthropic_count_tokens_501_envelope"] = ct501.status_code == 501 and _shape(
        ct501.json(), _SCHEMA_ANTH_ERROR
    )

    aml = client.get(_MODELS_PATH, headers=_ANTHROPIC_H)
    out["anthropic_models_list_schema"] = aml.status_code == 200 and _shape(
        aml.json(), _SCHEMA_ANTH_MODEL_LIST
    )
    amg = client.get(f"{_MODELS_PATH}/fx1", headers=_ANTHROPIC_H)
    out["anthropic_model_get_schema"] = amg.status_code == 200 and _shape(
        amg.json(), _SCHEMA_ANTH_MODEL
    )

    ab = client.post(
        f"{_MSGS_PATH}/batches",
        json={
            "requests": [
                {
                    "custom_id": "r1",
                    "params": {
                        "model": "fx1",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "x"}],
                    },
                }
            ]
        },
    )
    out["anthropic_batch_create_schema"] = ab.status_code == 200 and _shape(
        ab.json(), _SCHEMA_ANTH_BATCH
    )
    term: dict[str, Any] = {}
    for _ in range(500):
        term = dict(client.get(f"{_MSGS_PATH}/batches/{ab.json()['id']}").json())
        if term.get("processing_status") == "ended":
            break
        time.sleep(0.01)
    out["anthropic_batch_terminal_schema"] = term.get("processing_status") == "ended" and _shape(
        term, _SCHEMA_ANTH_BATCH
    )


def _probe_negative(out: dict[str, Any]) -> None:
    """Each surface must not leak the other vendor's grammar."""
    client, _ = _client(backend_resolver=lambda *a, **k: _StubBackend())
    metered, _ = _client(backend_resolver=lambda *a, **k: _MeterBackend())

    chat = client.post(
        _CHAT_PATH, json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]}
    ).json()
    chat_err = client.get(f"{_RESP_PATH}/resp_nope").json()
    out["neg_openai_no_anthropic_discriminator"] = (
        chat.get("type") != "error" and chat_err.get("type") != "error"
    )
    am = client.post(
        _MSGS_PATH,
        json={
            "model": "fx1",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
        },
    ).json()
    out["neg_anthropic_no_openai_keys"] = (
        "object" not in am
        and "choices" not in am
        and "created" not in am
        and "finish_reason" not in am
    )
    aerr = client.post(_MSGS_PATH, json={"model": "fx1", "max_tokens": 8}).json()
    out["neg_anthropic_err_no_openai_params"] = "param" not in aerr.get(
        "error", {}
    ) and "code" not in aerr.get("error", {})
    oai_usage = metered.post(
        _CHAT_PATH, json={"model": "fx1", "messages": [{"role": "user", "content": "x"}]}
    ).json()["usage"]
    resp_usage = metered.post(_RESP_PATH, json={"model": "fx1", "input": "x"}).json()["usage"]
    out["neg_usage_keys_never_cross"] = (
        not ({"input_tokens", "output_tokens"} & set(oai_usage))
        and not ({"prompt_tokens", "completion_tokens"} & set(resp_usage))
        and not ({"prompt_tokens", "completion_tokens", "total_tokens"} & set(am["usage"]))
    )
    ml = client.get(_MODELS_PATH).json()
    aml = client.get(_MODELS_PATH, headers=_ANTHROPIC_H).json()
    out["neg_model_lists_vendor_pure"] = (
        "first_id" not in ml and "last_id" not in ml and "object" not in aml
    )


def _probe_headers(out: dict[str, Any]) -> None:
    """429/401/403 header + envelope contracts under a keyed app."""
    keys_client, _ = _client(api_key=_ROOT_KEY, backend_resolver=lambda *a, **k: _StubBackend())
    root_h = {"X-API-Key": _ROOT_KEY}
    chat_body = {"model": "fx1", "messages": [{"role": "user", "content": "x"}]}
    msg_body = {"model": "fx1", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]}

    rpm_raw = str(keys_client.post("/harness/keys", json={"rpm": 1}, headers=root_h).json()["key"])
    rpm_h = {"X-API-Key": rpm_raw}
    keys_client.get("/harness/commands", headers=rpm_h)
    limited = keys_client.post(_CHAT_PATH, json=chat_body, headers=rpm_h)
    out["hdr_rpm_429_headers"] = (
        limited.status_code == 429
        and int(limited.headers.get("Retry-After", "0")) >= 1
        and limited.headers.get("x-ratelimit-limit-requests") == "1"
        and "x-ratelimit-remaining-requests" in limited.headers
        and "x-ratelimit-reset-requests" in limited.headers
    )
    out["hdr_rpm_429_envelope"] = _shape(limited.json(), _SCHEMA_OAI_ERROR)

    quota_raw = str(
        keys_client.post("/harness/keys", json={"max_requests": 1}, headers=root_h).json()["key"]
    )
    quota_h = {"X-API-Key": quota_raw}
    keys_client.get("/harness/commands", headers=quota_h)
    over = keys_client.post(_CHAT_PATH, json=chat_body, headers=quota_h)
    out["hdr_quota_429_no_retry_after"] = (
        over.status_code == 429
        and over.json()["error"]["code"] == "quota_exceeded"
        and "retry-after" not in {k.lower() for k in over.headers}
    )
    out["hdr_quota_429_envelope"] = _shape(over.json(), _SCHEMA_OAI_ERROR)
    over_a = keys_client.post(_MSGS_PATH, json=msg_body, headers=quota_h)
    out["hdr_quota_429_anthropic"] = (
        over_a.status_code == 429
        and _shape(over_a.json(), _SCHEMA_ANTH_ERROR)
        and "retry-after" not in {k.lower() for k in over_a.headers}
    )

    bad_h = {"X-API-Key": "k3y-wrong"}
    e401 = keys_client.post(_CHAT_PATH, json=chat_body, headers=bad_h)
    out["hdr_401_openai_envelope"] = e401.status_code == 401 and _shape(
        e401.json(), _SCHEMA_OAI_ERROR
    )
    e401a = keys_client.post(_MSGS_PATH, json=msg_body, headers=bad_h)
    out["hdr_401_anthropic_envelope"] = e401a.status_code == 401 and _shape(
        e401a.json(), _SCHEMA_ANTH_ERROR
    )

    ro_raw = str(
        keys_client.post("/harness/keys", json={"scopes": ["read"]}, headers=root_h).json()["key"]
    )
    ro_h = {"X-API-Key": ro_raw}
    e403 = keys_client.post(_CHAT_PATH, json=chat_body, headers=ro_h)
    out["hdr_403_openai_envelope"] = e403.status_code == 403 and _shape(
        e403.json(), _SCHEMA_OAI_ERROR
    )
    e403a = keys_client.post(_MSGS_PATH, json=msg_body, headers=ro_h)
    out["hdr_403_anthropic_envelope"] = e403a.status_code == 403 and _shape(
        e403a.json(), _SCHEMA_ANTH_ERROR
    )


def spec_audit() -> dict[str, Any]:
    """Run the vendor-spec battery — every probe a literal bool."""
    _SHAPE_ERRORS.clear()
    out: dict[str, Any] = {}
    _probe_chat(out)
    _probe_responses(out)
    _probe_models(out)
    _probe_legacy(out)
    _probe_files_batches(out)
    _probe_finetune(out)
    _probe_anthropic(out)
    _probe_negative(out)
    _probe_headers(out)
    return out


def spec_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under spec_audit.v1."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = spec_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "spec_audit",
        "schema": "spec_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "probes": len(r)},
        "interpretation": (
            "The /v1 wire speaks the vendors' grammars, not just our own "
            "OpenAPI export: chat completions + SSE chunks, responses "
            "(status enum, typed output, store-gated retrieval, "
            "previous_response_id chains), model list/retrieve, legacy "
            "text completions, file/batch/fine-tuning job objects all "
            "validate against the vendored OpenAI schema subset; the "
            "Anthropic surface (messages, SSE event grammar, "
            "count_tokens, cursor-paged models, message batches) "
            "validates against the vendored Anthropic subset; errors "
            "stay inside each vendor's envelope — OpenAI "
            "{error:{message,type,param,code}} never carries "
            "type:error, Anthropic {type:error,error:{type,message}} "
            "never carries param/code; and the 429/401/403 header "
            "contract holds (Retry-After + X-RateLimit-* on rpm limits, "
            "neither on quota_exceeded)."
            if ok
            else f"SPEC AUDIT DEFECT: {r} shape_errors={_SHAPE_ERRORS}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(spec_audit_bench(), indent=2, sort_keys=True))
