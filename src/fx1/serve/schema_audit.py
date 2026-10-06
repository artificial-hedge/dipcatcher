"""Request-schema/validation audit for the fx-1 serve harness — lane 164.

CLAIM UNDER TEST — every request model's declared validation edges are
real boundaries, exercised end to end over the wire grammar of each
surface: OpenAI-compatible ``/v1/*`` (chat completions, responses,
messages, legacy completions, embeddings, files, uploads, batches,
vector stores, conversations, evals/runs, fine-tuning), Anthropic
``/v1/messages*``, and the native ``/harness/*`` contract.

The battery pins the *measured* strictness policy honestly rather than
the caller's expectation:

- *type strictness* — pydantic lax coercion is the pinned policy:
  parseable scalars coerce (``"0.7"``→float, ``"2"``→int,
  ``"yes"``/``True``→bool/number), genuinely wrong types refuse
  ``422`` (``"hot"``, ``[0.5]``, ``1.5`` for an int, ``{}`` for a
  string-or-list field).
- *bounds* — every declared ``ge``/``le``/``min_length``/``max_length``
  refuses at its boundary and accepts at the cap; ``n`` caps at 8 on
  this harness (not OpenAI's 128), the request body cap is 1MiB (413),
  and vector/metadata caps pin exactly.
- *enums* — ``Literal`` fields refuse every out-of-domain value; the
  ``role`` field's strictness is surface-dependent (chat/harness
  free-form → accepted; responses → 400 at translation; Anthropic
  ``Literal`` → 422).
- *structure* — required fields, alternation, tool-call role scoping,
  union shapes, and nested-model errors all land the declared refusal.
- *cross-field* — ``max_tokens`` vs ``max_completion_tokens`` must
  agree; ``top_logprobs`` requires ``logprobs``; ``tool_choice``
  needs ``tools``; ``previous_response_id`` and ``conversation``
  exclude each other; ``background`` requires ``store``; ``stream``
  takes documented precedence over ``background``.
- *metadata* — chat/batch pin 16 keys / 64-char keys / 512-char values;
  vector stores refuse >16 at route level (400); eval-spec metadata is
  honestly unbounded (accepted at 17 keys — pinned, not laundered).
- *extra fields* — ``extra="allow"`` models swallow unknown keys
  (OpenAI-compatible loose binding); ``extra="forbid"`` models refuse
  ``422 extra_forbidden``, including forged ``_served_model``.
- *content edges* — control chars, NUL, unicode, and whitespace-only
  strings are accepted verbatim; the 1MiB body cap refuses 413;
  arbitrary nesting inside ``extra=allow`` fields is accepted.
- *batch lines* — a field-level violation in one batch line lands a
  per-line errored row (the batch still completes), while shape-level
  violations refuse the whole submit; ``stream``/``background``/
  ``conversation`` lines refuse per-line.
- *error shape* — harness 422s carry ``{detail, code: "validation"}``;
  ``/v1`` refusals carry the OpenAI or Anthropic ``error`` envelope;
  no refusal anywhere is a bare 500.
- *validity invariant* — a refused request burns no backend call,
  stores no record, and logs no served-call, though the credential's
  ``uses`` counter honestly bills the auth'd attempt.

This is a stub-backed TestClient battery — transport is in-process;
provider-side validation beyond this grammar is out of scope.

Sealed ``schema_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fx1.serve.backends import EmbeddingResult
from fx1.serve.conv_audit import (
    _audit_context,
    _client,
    _msg,
    _StubBackend,
)
from fx1.serve.perf_audit import _err_code, _wait_for
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["schema_audit", "schema_audit_bench"]

_ROOT = "schema-aud1t-r00t"
_MODEL = "hosted_k3"
_H: dict[str, str] = {"X-API-Key": _ROOT}

_CHAT = "/v1/chat/completions"
_MSG = "/v1/messages"
_MSG_BATCH = "/v1/messages/batches"
_MSG_COUNT = "/v1/messages/count_tokens"
_RESP = "/v1/responses"
_LEGACY = "/v1/completions"
_EMB = "/v1/embeddings"
_MOD = "/v1/moderations"
_FILES = "/v1/files"
_UPLOADS = "/v1/uploads"
_BATCHES = "/v1/batches"
_VS = "/v1/vector_stores"
_CONV = "/v1/conversations"
_EVALS = "/v1/evals"
_FT = "/v1/fine_tuning/jobs"

_COMPLETE = "/harness/complete"
_COMPLETE_STREAM = "/harness/complete/stream"
_COMPLETE_BATCH = "/harness/complete/batch"
_JOBS = "/harness/jobs"
_JOBS_BATCH = "/harness/jobs/batch"
_EVALS_H = "/harness/evals"
_GATE = "/harness/gate/check"
_SCORE = "/harness/score"
_PROBE = "/harness/backends/hosted_k3/probe"
_KEYS = "/harness/keys"
_SELF = "/harness/self"
_VERIFY = "/receipts/verify"
_VERIFY_BATCH = "/receipts/verify/batch"

# Every (path, status, error-code) the battery observes — the global
# error-shape invariants at the end read this ledger, so a refusal that
# ever lands a bare 500 or an undeclared 5xx code flips the pin no
# matter which family triggered it.
_SEEN: list[tuple[str, int, str | None]] = []

# 5xx verdicts that are declared, not crashes: capability refusals
# (backend lacks the channel), output-contract refusals (model output
# isn't JSON under a declared format), saturation (over_capacity/
# draining). A refusal must never land a bare 500. The Anthropic
# surface carries its verdict token in ``error.type`` — ``api_error``
# and ``overloaded_error`` are its declared 5xx types.
_HONEST_5XX = frozenset(
    {
        "not_implemented",
        "not_supported",
        "format_violation",
        "backend_unavailable",
        "upstream_unavailable",
        "over_capacity",
        "draining",
        "timeout",
        "api_error",
        "overloaded_error",
    }
)


class _FullBackend(_StubBackend):
    """``_StubBackend`` plus the two optional capability channels, so the
    battery reaches the validated happy path on ``/v1/embeddings`` and
    ``/v1/messages/count_tokens`` instead of stopping at the 501
    capability verdict."""

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        del encoding_format, dimensions, user
        self.calls += 1
        return EmbeddingResult(
            data=({"object": "embedding", "index": 0, "embedding": [0.0]},),
            model=model,
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.calls += 1
        return 7


def _backends(shared: _FullBackend | None = None) -> dict[str, Any]:
    """backend_map — values are zero-arg factories; passing a shared
    ``shared`` stub keeps its call counter meaningful across requests."""

    def _hosted() -> Any:
        return shared if shared is not None else _FullBackend()

    return {"hosted_k3": _hosted, "local_fx1": _StubBackend, "byok": _StubBackend}


def _code_of(r: Any) -> str | None:
    """Machine-readable verdict token in either grammar — OpenAI/
    harness ``error.code`` (or bare ``code``), else the Anthropic
    envelope's ``error.type`` slot."""
    code = _err_code(r)
    if code is not None:
        return code
    try:
        err = r.json().get("error")
    except ValueError:
        return None
    if isinstance(err, dict):
        t = err.get("type")
        return str(t) if t is not None else None
    return None


def _note(path: str, r: Any) -> None:
    _SEEN.append((path, int(r.status_code), _code_of(r)))


def _req(
    client: TestClient,
    method: str,
    path: str,
    *,
    body: Any = ...,  # ellipsis sentinel — None is a real JSON body
    headers: dict[str, str] | None = None,
    raw: bytes | None = None,
    files: Any = None,
    data: Any = None,
) -> Any:
    """One instrumented call — every response lands in the ledger."""
    hdrs = {**_H, **(headers or {})}
    if raw is not None:
        hdrs.setdefault("Content-Type", "application/json")
        r = client.request(method, path, content=raw, headers=hdrs)
    elif files is not None or data is not None:
        r = client.request(method, path, files=files, data=data, headers=hdrs)
    elif body is ...:
        r = client.request(method, path, headers=hdrs)
    else:
        r = client.request(method, path, json=body, headers=hdrs)
    _note(path, r)
    return r


def _post(client: TestClient, path: str, body: Any = ..., **kw: Any) -> Any:
    return _req(client, "post", path, body=body, **kw)


def _get(client: TestClient, path: str, **kw: Any) -> Any:
    return _req(client, "get", path, **kw)


def _chat(content: Any = "x", **extra: Any) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def _msg_body(content: Any = "x", **extra: Any) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def _resp(input: Any = "x", **extra: Any) -> dict[str, Any]:  # noqa: A002
    return {"model": _MODEL, "input": input, **extra}


def _legacy(**extra: Any) -> dict[str, Any]:
    return {"model": _MODEL, "prompt": "x", **extra}


def _emb(**extra: Any) -> dict[str, Any]:
    return {"model": _MODEL, "input": "x", **extra}


def _complete(**extra: Any) -> dict[str, Any]:
    return {
        "backend": _MODEL,
        "messages": [{"role": "user", "content": "x"}],
        **extra,
    }


def _tools(n: int) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": f"f{i}", "parameters": {"type": "object"}}}
        for i in range(n)
    ]


def _msgs(n: int, *, alternate: bool = False) -> list[dict[str, Any]]:
    if not alternate:
        return [{"role": "user", "content": "x"}] * n
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "x"} for i in range(n)]


def _meta(n: int, *, val_len: int = 1) -> dict[str, str]:
    return {f"k{i:04d}": "v" * val_len for i in range(n)}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = _post(client, _KEYS, policy)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _evalspec(**extra: Any) -> dict[str, Any]:
    return {
        "name": "audit-spec",
        "data_source_config": {"type": "custom", "item_schema": {"suite": "capability"}},
        **extra,
    }


def _refusals(cases: tuple[tuple[str, str, str, Any], ...], client: TestClient) -> dict[str, bool]:
    """Every case must land exactly 422."""
    return {
        name: _req(client, method, path, body=body).status_code == 422
        for name, method, path, body in cases
    }


# ---------------------------------------------------------------------------
# Type strictness — the measured coercion policy
# ---------------------------------------------------------------------------


def _type_strictness_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out = _refusals(
        (
            ("chat_temp_word", "post", _CHAT, _chat(temperature="hot")),
            ("chat_temp_list", "post", _CHAT, _chat(temperature=[0.5])),
            ("chat_max_tokens_float", "post", _CHAT, _chat(max_tokens=1.5)),
            ("chat_n_word", "post", _CHAT, _chat(n="abc")),
            ("chat_stop_int", "post", _CHAT, _chat(stop=5)),
            ("chat_stop_item_int", "post", _CHAT, _chat(stop=["a", 1])),
            ("chat_messages_elem_str", "post", _CHAT, {"model": _MODEL, "messages": ["hi"]}),
            ("chat_content_int", "post", _CHAT, _chat(5)),
            ("harness_temp_word", "post", _COMPLETE, _complete(temperature="hot")),
            (
                "harness_stream_route_same_grammar",
                "post",
                _COMPLETE_STREAM,
                _complete(temperature="hot"),
            ),
            ("harness_max_tokens_float", "post", _COMPLETE, _complete(max_tokens=1.5)),
            (
                "harness_messages_elem_str",
                "post",
                _COMPLETE,
                {"backend": _MODEL, "messages": ["hi"]},
            ),
            ("resp_input_int", "post", _RESP, _resp(5)),
            ("resp_input_dict", "post", _RESP, _resp({})),
            ("resp_reasoning_str", "post", _RESP, _resp(reasoning="low")),
            ("msg_max_tokens_float", "post", _MSG, _msg_body(max_tokens=1.5)),
            (
                "msg_max_tokens_missing",
                "post",
                _MSG,
                {"model": _MODEL, "messages": [{"role": "user", "content": "x"}]},
            ),
            ("legacy_prompt_int", "post", _LEGACY, _legacy(prompt=5)),
            ("legacy_prompt_item_int", "post", _LEGACY, _legacy(prompt=["ok", 3])),
            ("emb_input_dict", "post", _EMB, _emb(input={})),
            ("emb_input_int", "post", _EMB, _emb(input=5)),
            ("emb_input_mixed_types", "post", _EMB, _emb(input=["x", [1, 2]])),
            ("jobs_command_int", "post", _JOBS, {"command": 123}),
            (
                "jobs_extra_args_elem_int",
                "post",
                _JOBS,
                {"command": "backtest", "extra_args": ["a", 1]},
            ),
            ("key_name_int", "post", _KEYS, {"name": 5}),
            (
                "upload_bytes_float",
                "post",
                _UPLOADS,
                {"purpose": "batch", "filename": "f.jsonl", "bytes": 1.5, "mime_type": "text"},
            ),
        ),
        client,
    )
    out = dict(out)
    # Lax coercion is the pinned policy — parseable scalars coerce, and
    # bools are ints/floats under lax mode. Honest pins, not wishes.
    out["chat_temp_string_coerced"] = (
        _post(client, _CHAT, _chat(temperature="0.7")).status_code == 200
    )
    out["chat_temp_bool_coerced"] = _post(client, _CHAT, _chat(temperature=True)).status_code == 200
    out["chat_max_tokens_bool_coerced"] = (
        _post(client, _CHAT, _chat(max_tokens=True)).status_code == 200
    )
    n2 = _post(client, _CHAT, _chat(n="2"))
    out["chat_n_string_coerced"] = n2.status_code == 200 and len(n2.json().get("choices", [])) == 2
    r = _post(client, _CHAT, _chat(stream="yes"))
    out["chat_stream_word_coerced_sse"] = r.status_code == 200 and "data:" in r.text
    out["resp_input_string_ok"] = _post(client, _RESP, _resp("x")).status_code == 200
    out["resp_temp_bool_coerced"] = _post(client, _RESP, _resp(temperature=True)).status_code == 200
    out["msg_temp_bool_coerced"] = (
        _post(client, _MSG, _msg_body(temperature=True)).status_code == 200
    )
    out["msg_temp_string_coerced"] = (
        _post(client, _MSG, _msg_body(temperature="0.5")).status_code == 200
    )
    out["upload_bytes_bool_coerced"] = (
        _post(
            client,
            _UPLOADS,
            {"purpose": "batch", "filename": "f.jsonl", "bytes": True, "mime_type": "text"},
        ).status_code
        == 200
    )
    eval_id = _eval_id(client)
    out["evalrun_model_int_refused"] = (
        _post(client, f"{_EVALS}/{eval_id}/runs", {"model": 5}).status_code == 422
    )
    return out


def _eval_id(client: TestClient) -> str:
    r = _post(client, _EVALS, _evalspec())
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


# ---------------------------------------------------------------------------
# Bounds — every declared cap refuses past the boundary, accepts at it
# ---------------------------------------------------------------------------


def _bounds_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    out.update(
        _refusals(
            (
                ("chat_temp_below_zero", "post", _CHAT, _chat(temperature=-0.1)),
                ("chat_temp_above_two", "post", _CHAT, _chat(temperature=2.1)),
                ("chat_top_p_zero", "post", _CHAT, _chat(top_p=0)),
                ("chat_top_p_above", "post", _CHAT, _chat(top_p=1.01)),
                ("chat_max_tokens_neg", "post", _CHAT, _chat(max_tokens=-1)),
                ("chat_max_tokens_zero", "post", _CHAT, _chat(max_tokens=0)),
                ("chat_max_tokens_overflow", "post", _CHAT, _chat(max_tokens=10**20)),
                ("chat_n_zero", "post", _CHAT, _chat(n=0)),
                ("chat_n_above_cap", "post", _CHAT, _chat(n=9)),
                ("chat_seed_neg", "post", _CHAT, _chat(seed=-1)),
                ("chat_stop_above_cap", "post", _CHAT, _chat(stop=["a"] * 5)),
                ("chat_stop_item_long", "post", _CHAT, _chat(stop=["x" * 513])),
                ("chat_freq_penalty_high", "post", _CHAT, _chat(frequency_penalty=2.1)),
                ("chat_pres_penalty_low", "post", _CHAT, _chat(presence_penalty=-2.1)),
                ("chat_top_logprobs_neg", "post", _CHAT, _chat(logprobs=True, top_logprobs=-1)),
                ("chat_top_logprobs_above", "post", _CHAT, _chat(logprobs=True, top_logprobs=21)),
                ("chat_logit_bias_range", "post", _CHAT, _chat(logit_bias={"1": 101})),
                (
                    "chat_model_empty",
                    "post",
                    _CHAT,
                    {"model": "", "messages": [{"role": "user", "content": "x"}]},
                ),
                (
                    "chat_model_over",
                    "post",
                    _CHAT,
                    {"model": "m" * 513, "messages": [{"role": "user", "content": "x"}]},
                ),
                ("chat_user_over", "post", _CHAT, _chat(user="u" * 513)),
                ("chat_msgs_over", "post", _CHAT, {"model": _MODEL, "messages": _msgs(513)}),
                ("chat_tools_over", "post", _CHAT, _chat(tools=_tools(129))),
                (
                    "chat_tool_name_over",
                    "post",
                    _CHAT,
                    _chat(tools=[{"type": "function", "function": {"name": "f" * 65}}]),
                ),
                (
                    "chat_tool_name_badchar",
                    "post",
                    _CHAT,
                    _chat(tools=[{"type": "function", "function": {"name": "bad name!"}}]),
                ),
                (
                    "chat_tool_desc_over",
                    "post",
                    _CHAT,
                    _chat(
                        tools=[
                            {
                                "type": "function",
                                "function": {"name": "f", "description": "d" * 8193},
                            }
                        ]
                    ),
                ),
                ("resp_max_output_tokens_zero", "post", _RESP, _resp(max_output_tokens=0)),
                ("resp_max_output_tokens_over", "post", _RESP, _resp(max_output_tokens=262145)),
                ("resp_max_tool_calls_neg", "post", _RESP, _resp(max_tool_calls=-1)),
                ("resp_instructions_over", "post", _RESP, _resp(instructions="i" * 32769)),
                ("resp_tools_over", "post", _RESP, _resp(tools=_tools(129))),
                ("resp_input_over", "post", _RESP, _resp([{"role": "user", "content": "x"}] * 513)),
                (
                    "resp_fs_stores_over",
                    "post",
                    _RESP,
                    _resp(
                        tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": [f"vs_{i}" for i in range(9)],
                            }
                        ]
                    ),
                ),
                (
                    "resp_fs_stores_empty",
                    "post",
                    _RESP,
                    _resp(tools=[{"type": "file_search", "vector_store_ids": []}]),
                ),
                (
                    "resp_fs_results_over",
                    "post",
                    _RESP,
                    _resp(
                        tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": ["vs_a"],
                                "max_num_results": 51,
                            }
                        ]
                    ),
                ),
                (
                    "resp_fs_threshold_low",
                    "post",
                    _RESP,
                    _resp(
                        tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": ["vs_a"],
                                "ranking_options": {"ranker": "auto", "score_threshold": -0.1},
                            }
                        ]
                    ),
                ),
                (
                    "resp_fs_threshold_high",
                    "post",
                    _RESP,
                    _resp(
                        tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": ["vs_a"],
                                "ranking_options": {"ranker": "auto", "score_threshold": 1.1},
                            }
                        ]
                    ),
                ),
                ("msg_max_tokens_zero", "post", _MSG, _msg_body(max_tokens=0)),
                ("msg_max_tokens_over", "post", _MSG, _msg_body(max_tokens=262145)),
                ("msg_temp_above_one", "post", _MSG, _msg_body(temperature=1.1)),
                ("msg_top_p_zero", "post", _MSG, _msg_body(top_p=0)),
                ("msg_stop_above", "post", _MSG, _msg_body(stop_sequences=["a"] * 5)),
                ("msg_msgs_over", "post", _MSG, _msg_body(messages=_msgs(513, alternate=True))),
                (
                    "msg_tools_over",
                    "post",
                    _MSG,
                    _msg_body(
                        tools=[
                            {"name": f"f{i}", "input_schema": {"type": "object"}}
                            for i in range(129)
                        ]
                    ),
                ),
                ("legacy_prompt_over", "post", _LEGACY, _legacy(prompt="x" * 131073)),
                ("legacy_prompt_items_over", "post", _LEGACY, _legacy(prompt=["x"] * 513)),
                ("legacy_n_zero", "post", _LEGACY, _legacy(n=0)),
                ("legacy_n_over", "post", _LEGACY, _legacy(n=9)),
                ("harness_temp_above", "post", _COMPLETE, _complete(temperature=2.1)),
                ("harness_top_p_zero", "post", _COMPLETE, _complete(top_p=0)),
                (
                    "harness_fallbacks_over",
                    "post",
                    _COMPLETE,
                    _complete(fallbacks=["local_fx1", "byok", "hosted_k3"]),
                ),
                ("harness_user_over", "post", _COMPLETE, _complete(user="u" * 513)),
                (
                    "harness_msgs_over",
                    "post",
                    _COMPLETE,
                    {"backend": _MODEL, "messages": _msgs(513)},
                ),
                ("harness_tools_over", "post", _COMPLETE, _complete(tools=_tools(129))),
                (
                    "harness_checkpoint_nul",
                    "post",
                    _COMPLETE,
                    {
                        "backend": "local_fx1",
                        "messages": [{"role": "user", "content": "x"}],
                        "checkpoint_dir": "a\x00b",
                    },
                ),
                ("harness_stop_items_over", "post", _COMPLETE, _complete(stop=["a"] * 5)),
                ("harness_stop_item_over", "post", _COMPLETE, _complete(stop=["x" * 513])),
                (
                    "cb_batch_over",
                    "post",
                    _COMPLETE_BATCH,
                    {"backend": _MODEL, "batch": [_msgs(1)] * 65},
                ),
                ("cb_batch_empty", "post", _COMPLETE_BATCH, {"backend": _MODEL, "batch": []}),
                (
                    "cb_workers_over",
                    "post",
                    _COMPLETE_BATCH,
                    {"backend": _MODEL, "batch": [_msgs(1)], "max_workers": 17},
                ),
                (
                    "cb_workers_zero",
                    "post",
                    _COMPLETE_BATCH,
                    {"backend": _MODEL, "batch": [_msgs(1)], "max_workers": 0},
                ),
                ("jobs_batch_over", "post", _JOBS_BATCH, {"jobs": [{"command": "backtest"}] * 65}),
                ("jobs_batch_empty", "post", _JOBS_BATCH, {"jobs": []}),
                (
                    "jobs_extra_args_over",
                    "post",
                    _JOBS,
                    {"command": "backtest", "extra_args": ["a"] * 65},
                ),
                (
                    "jobs_idemkey_over",
                    "post",
                    _JOBS,
                    {"command": "backtest", "idempotency_key": "k" * 257},
                ),
                ("gate_text_over", "post", _GATE, {"text": "x" * 262145}),
                ("score_items_over", "post", _SCORE, {"input": ["x"] * 129}),
                ("score_item_over", "post", _SCORE, {"input": ["x" * 262145]}),
                ("score_empty_list", "post", _SCORE, {"input": []}),
                ("key_name_over", "post", _KEYS, {"name": "k" * 129}),
                ("key_rpm_zero", "post", _KEYS, {"rpm": 0}),
                ("key_rpm_over", "post", _KEYS, {"rpm": 1_000_001}),
                ("key_ttl_zero", "post", _KEYS, {"ttl_s": 0}),
                ("key_ttl_over", "post", _KEYS, {"ttl_s": 315_576_001}),
                ("key_max_requests_zero", "post", _KEYS, {"max_requests": 0}),
                ("key_max_requests_over", "post", _KEYS, {"max_requests": 2**31}),
                ("key_max_tokens_zero", "post", _KEYS, {"max_tokens": 0}),
                ("key_max_tokens_over", "post", _KEYS, {"max_tokens": 2**63}),
                (
                    "upload_purpose_over",
                    "post",
                    _UPLOADS,
                    {"purpose": "p" * 33, "filename": "f.jsonl", "bytes": 4, "mime_type": "t"},
                ),
                (
                    "upload_filename_over",
                    "post",
                    _UPLOADS,
                    {"purpose": "batch", "filename": "f" * 257, "bytes": 4, "mime_type": "t"},
                ),
                (
                    "upload_bytes_zero",
                    "post",
                    _UPLOADS,
                    {"purpose": "batch", "filename": "f.jsonl", "bytes": 0, "mime_type": "t"},
                ),
                (
                    "upload_mime_over",
                    "post",
                    _UPLOADS,
                    {"purpose": "batch", "filename": "f.jsonl", "bytes": 4, "mime_type": "m" * 129},
                ),
                ("verify_batch_over", "post", _VERIFY_BATCH, {"receipts": [{}] * 65}),
                ("verify_batch_empty", "post", _VERIFY_BATCH, {"receipts": []}),
                (
                    "vs_create_file_ids_over",
                    "post",
                    _VS,
                    {"name": "s", "file_ids": [f"file-{i}" for i in range(65)]},
                ),
                (
                    "espec_criteria_over",
                    "post",
                    _EVALS,
                    _evalspec(testing_criteria=[{"name": f"c{i}"} for i in range(65)]),
                ),
                ("emb_input_over", "post", _EMB, _emb(input=["x"] * 2049)),
                ("emb_dims_zero", "post", _EMB, _emb(dimensions=0)),
                (
                    "msgb_custom_id_over",
                    "post",
                    _MSG_BATCH,
                    {"requests": [{"custom_id": "c" * 257, "params": _msg_body()}]},
                ),
            ),
            client,
        )
    )
    # Cap-boundary acceptance — the declared cap itself must still serve.
    out["chat_temp_edge_two_ok"] = _post(client, _CHAT, _chat(temperature=2.0)).status_code == 200
    out["chat_top_p_edge_one_ok"] = _post(client, _CHAT, _chat(top_p=1.0)).status_code == 200
    out["chat_max_tokens_cap_ok"] = (
        _post(client, _CHAT, _chat(max_tokens=262144)).status_code == 200
    )
    n8 = _post(client, _CHAT, _chat(n=8))
    out["chat_n_cap_ok"] = n8.status_code == 200 and len(n8.json().get("choices", [])) == 8
    out["chat_penalty_edge_ok"] = (
        _post(client, _CHAT, _chat(frequency_penalty=2.0, presence_penalty=-2.0)).status_code == 200
    )
    # logprobs:true itself reaches the structured-completion channel —
    # validation accepts it, then the capability verdict lands honestly.
    tlp = _post(client, _CHAT, _chat(logprobs=True, top_logprobs=20))
    out["chat_top_logprobs_cap_validated_then_501"] = tlp.status_code == 501
    out["chat_logit_bias_edge_ok"] = (
        _post(client, _CHAT, _chat(logit_bias={"1": 100, "-1": -100})).status_code == 200
    )
    out["chat_seed_large_ok"] = _post(client, _CHAT, _chat(seed=10**18)).status_code == 200
    out["chat_msgs_cap_ok"] = (
        _post(client, _CHAT, {"model": _MODEL, "messages": _msgs(512)}).status_code == 200
    )
    tools128 = _post(client, _CHAT, _chat(tools=_tools(128)))
    out["chat_tools_cap_validated_then_501"] = tools128.status_code == 501
    out["resp_input_cap_ok"] = (
        _post(client, _RESP, _resp([{"role": "user", "content": "x"}] * 512)).status_code == 200
    )
    out["msg_max_tokens_cap_ok"] = (
        _post(client, _MSG, _msg_body(max_tokens=262144)).status_code == 200
    )
    out["msg_msgs_cap_ok"] = (
        _post(client, _MSG, _msg_body(messages=_msgs(512, alternate=True))).status_code == 200
    )
    msg_tools128 = _post(
        client,
        _MSG,
        _msg_body(
            tools=[{"name": f"f{i}", "input_schema": {"type": "object"}} for i in range(128)]
        ),
    )
    out["msg_tools_cap_validated_then_501"] = msg_tools128.status_code == 501
    out["legacy_prompt_edge_ok"] = (
        _post(client, _LEGACY, _legacy(prompt="x" * 131072)).status_code == 200
    )
    out["legacy_n_cap_ok"] = _post(client, _LEGACY, _legacy(n=8)).status_code == 200
    # DEFECT FIXED THIS LANE: an empty inner conversation used to reach
    # ``backend.complete([])`` and crash the call site (500). The model
    # now requires each inner list non-empty — same contract as
    # ``messages: []`` on the single-turn route.
    out["cb_inner_empty_422"] = (
        _post(client, _COMPLETE_BATCH, {"backend": _MODEL, "batch": [[]]}).status_code == 422
    )
    out["cb_inner_empty_mixed_422"] = (
        _post(client, _COMPLETE_BATCH, {"backend": _MODEL, "batch": [[], _msgs(1)]}).status_code
        == 422
    )
    out["gate_text_edge_ok"] = _post(client, _GATE, {"text": "x" * 262144}).status_code == 200
    out["score_items_cap_ok"] = _post(client, _SCORE, {"input": ["x"] * 128}).status_code == 200
    out["score_empty_str_ok"] = _post(client, _SCORE, {"input": ""}).status_code == 200
    out["emb_input_cap_ok"] = _post(client, _EMB, _emb(input=["x"] * 2048)).status_code == 200
    out["espec_criteria_cap_ok"] = (
        _post(
            client, _EVALS, _evalspec(testing_criteria=[{"name": f"c{i}"} for i in range(64)])
        ).status_code
        == 201
    )
    return out


# ---------------------------------------------------------------------------
# Enums, literals, and field-level grammars
# ---------------------------------------------------------------------------


def _enum_literal_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    out.update(
        _refusals(
            (
                ("chat_rf_bogus", "post", _CHAT, _chat(response_format={"type": "bogus"})),
                (
                    "chat_rf_json_schema_no_schema",
                    "post",
                    _CHAT,
                    _chat(response_format={"type": "json_schema", "json_schema": {"name": "s"}}),
                ),
                (
                    "chat_rf_json_schema_bad_schema",
                    "post",
                    _CHAT,
                    _chat(
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "s", "schema": {"type": "bogus-type"}},
                        }
                    ),
                ),
                ("chat_tool_choice_str_bogus", "post", _CHAT, _chat(tool_choice="bogus")),
                (
                    "chat_tool_choice_type_bogus",
                    "post",
                    _CHAT,
                    _chat(tools=_tools(1), tool_choice={"type": "bogus"}),
                ),
                ("chat_reasoning_ultra", "post", _CHAT, _chat(reasoning_effort="ultra")),
                ("chat_service_tier_bogus", "post", _CHAT, _chat(service_tier="bogus")),
                ("chat_verbosity_bogus", "post", _CHAT, _chat(verbosity="bogus")),
                ("chat_pcr_bogus", "post", _CHAT, _chat(prompt_cache_retention="bogus")),
                (
                    "chat_tool_type_nonfunction",
                    "post",
                    _CHAT,
                    _chat(tools=[{"type": "other", "function": {"name": "f"}}]),
                ),
                (
                    "msg_role_sysadmin",
                    "post",
                    _MSG,
                    _msg_body(messages=[{"role": "sysadmin", "content": "x"}]),
                ),
                (
                    "msg_tool_choice_type_bogus",
                    "post",
                    _MSG,
                    _msg_body(
                        tools=[{"name": "f", "input_schema": {"type": "object"}}],
                        tool_choice={"type": "bogus"},
                    ),
                ),
                (
                    "msg_tool_choice_name_on_none",
                    "post",
                    _MSG,
                    _msg_body(tool_choice={"type": "none", "name": "f"}),
                ),
                (
                    "resp_reasoning_effort_ultra",
                    "post",
                    _RESP,
                    _resp(reasoning={"effort": "ultra"}),
                ),
                (
                    "resp_reasoning_extra_key",
                    "post",
                    _RESP,
                    _resp(reasoning={"effort": "low", "bogus": 1}),
                ),
                ("resp_text_verbosity_bogus", "post", _RESP, _resp(text={"verbosity": "bogus"})),
                ("resp_include_bogus", "post", _RESP, _resp(include=["bogus"])),
                ("resp_tool_choice_bogus", "post", _RESP, _resp(tool_choice="bogus")),
                ("emb_encoding_bogus", "post", _EMB, _emb(encoding_format="bogus")),
                (
                    "batch_endpoint_bogus",
                    "post",
                    _BATCHES,
                    {"input_file_id": "file-x", "endpoint": "/v1/nope", "completion_window": "24h"},
                ),
                (
                    "batch_window_bogus",
                    "post",
                    _BATCHES,
                    {"input_file_id": "file-x", "endpoint": _CHAT, "completion_window": "48h"},
                ),
                (
                    "ft_method_bogus",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "method": {"type": "bogus"}},
                ),
                (
                    "ft_suffix_badchar",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "suffix": "BAD SUFFIX"},
                ),
                (
                    "ft_epochs_zero",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "hyperparameters": {"n_epochs": 0}},
                ),
                (
                    "ft_epochs_over",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "hyperparameters": {"n_epochs": 51}},
                ),
                (
                    "ft_lr_zero",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "hyperparameters": {"lr_mult": 0}},
                ),
                (
                    "ft_lr_over",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "hyperparameters": {"lr_mult": 11}},
                ),
                (
                    "ft_batch_over",
                    "post",
                    _FT,
                    {
                        "model": "m",
                        "training_file": "file-x",
                        "hyperparameters": {"batch_size": 513},
                    },
                ),
                ("evalsubmit_suite_bogus", "post", _EVALS_H, {"suite": "bogus", "backend": _MODEL}),
                (
                    "evalsubmit_backend_bogus",
                    "post",
                    _EVALS_H,
                    {"suite": "capability", "backend": "bogus"},
                ),
                (
                    "espec_suite_bogus",
                    "post",
                    _EVALS,
                    {
                        "name": "e",
                        "data_source_config": {"type": "custom", "item_schema": {"suite": "bogus"}},
                    },
                ),
                (
                    "espec_type_bogus",
                    "post",
                    _EVALS,
                    {
                        "name": "e",
                        "data_source_config": {
                            "type": "bogus",
                            "item_schema": {"suite": "capability"},
                        },
                    },
                ),
                (
                    "espec_dsc_extra_key",
                    "post",
                    _EVALS,
                    {
                        "name": "e",
                        "data_source_config": {
                            "type": "custom",
                            "source": {"t": 1},
                            "item_schema": {"suite": "capability"},
                        },
                    },
                ),
                (
                    "msgb_stream_param",
                    "post",
                    _MSG_BATCH,
                    {"requests": [{"custom_id": "a", "params": _msg_body(stream=True)}]},
                ),
                ("fx1_backend_bogus", "post", _CHAT, _chat(fx1={"backend": "bogus"})),
                ("fx1_timeout_zero", "post", _CHAT, _chat(fx1={"timeout_s": 0})),
                (
                    "fx1_fallbacks_over",
                    "post",
                    _CHAT,
                    _chat(fx1={"fallbacks": ["byok", "local_fx1", "hosted_k3"]}),
                ),
                (
                    "fx1_byok_bad_url",
                    "post",
                    _CHAT,
                    _chat(
                        fx1={
                            "backend": "byok",
                            "byok": {"base_url": "ftp://x", "api_key": "k", "model": "m"},
                        }
                    ),
                ),
            ),
            client,
        )
    )
    # Role strictness is surface-dependent — pin each actual policy.
    out["chat_role_sysadmin_accepted"] = (
        _post(client, _CHAT, _chat(messages=[{"role": "sysadmin", "content": "x"}])).status_code
        == 200
    )
    out["resp_role_sysadmin_refused_400"] = (
        _post(client, _RESP, _resp([{"role": "sysadmin", "content": "x"}])).status_code == 400
    )
    out["harness_role_sysadmin_accepted"] = (
        _post(
            client,
            _COMPLETE,
            {"backend": _MODEL, "messages": [{"role": "sysadmin", "content": "x"}]},
        ).status_code
        == 200
    )
    # json_schema without a name is VALID at the model layer — the
    # refusal lands at output validation (502 format_violation), an
    # honest declared verdict on the stub's non-JSON echo.
    rf = _post(
        client,
        _CHAT,
        _chat(
            response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        ),
    )
    out["chat_rf_no_name_validates_then_output_502"] = (
        rf.status_code == 502 and _err_code(rf) == "format_violation"
    )
    out["chat_reasoning_high_ok"] = (
        _post(client, _CHAT, _chat(reasoning_effort="high")).status_code == 200
    )
    out["emb_encoding_base64_ok"] = (
        _post(client, _EMB, _emb(encoding_format="base64")).status_code == 200
    )
    _key, kid = _mint(client)
    out["key_rotate_ttl_zero"] = (
        _post(client, f"{_KEYS}/{kid}/rotate", {"ttl_s": 0}).status_code == 422
    )
    # Unknown scope names are refused route-side (400 scopes_invalid),
    # not at the model — the scope registry is a routing concern.
    out["key_scopes_bogus_400"] = (
        _post(client, _KEYS, {"scopes": ["superadmin"]}).status_code == 400
    )
    return out


# ---------------------------------------------------------------------------
# Structure — shapes, required fields, role scoping, alternation
# ---------------------------------------------------------------------------


def _structure_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    out.update(
        _refusals(
            (
                ("chat_messages_empty", "post", _CHAT, _chat(messages=[])),
                ("chat_messages_dict", "post", _CHAT, {"model": _MODEL, "messages": {}}),
                (
                    "resp_input_items_elem_str",
                    "post",
                    _RESP,
                    _resp([{"role": "user", "content": "x"}, "y"]),
                ),
                (
                    "msg_first_not_user",
                    "post",
                    _MSG,
                    _msg_body(
                        messages=[
                            {"role": "assistant", "content": "x"},
                            {"role": "user", "content": "y"},
                        ]
                    ),
                ),
                (
                    "msg_alternation_user_user",
                    "post",
                    _MSG,
                    _msg_body(
                        messages=[
                            {"role": "user", "content": "x"},
                            {"role": "user", "content": "y"},
                        ]
                    ),
                ),
                (
                    "msg_content_empty_list",
                    "post",
                    _MSG,
                    _msg_body(messages=[{"role": "user", "content": []}]),
                ),
                (
                    "msg_tool_use_under_user",
                    "post",
                    _MSG,
                    _msg_body(
                        messages=[
                            {"role": "user", "content": "x"},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "tool_use", "id": "t", "name": "f", "input": {}}
                                ],
                            },
                        ]
                    ),
                ),
                (
                    "msg_tool_result_under_assistant",
                    "post",
                    _MSG,
                    _msg_body(
                        messages=[
                            {"role": "user", "content": "x"},
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": "t", "content": "y"}
                                ],
                            },
                        ]
                    ),
                ),
                (
                    "msg_tools_dup_name",
                    "post",
                    _MSG,
                    _msg_body(
                        tools=[
                            {"name": "f", "input_schema": {"type": "object"}},
                            {"name": "f", "input_schema": {"type": "object"}},
                        ]
                    ),
                ),
                (
                    "msg_tool_schema_nonobject",
                    "post",
                    _MSG,
                    _msg_body(tools=[{"name": "f", "input_schema": {"type": "array"}}]),
                ),
                (
                    "msg_cache_control_refused",
                    "post",
                    _MSG,
                    _msg_body(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "x",
                                        "cache_control": {"type": "ephemeral"},
                                    }
                                ],
                            }
                        ]
                    ),
                ),
                (
                    "harness_tool_calls_under_user",
                    "post",
                    _COMPLETE,
                    {
                        "backend": _MODEL,
                        "messages": [
                            {
                                "role": "user",
                                "content": "x",
                                "tool_calls": [
                                    {
                                        "id": "t1",
                                        "type": "function",
                                        "function": {"name": "f", "arguments": "{}"},
                                    }
                                ],
                            }
                        ],
                    },
                ),
                (
                    "harness_tool_call_id_under_assistant",
                    "post",
                    _COMPLETE,
                    {
                        "backend": _MODEL,
                        "messages": [{"role": "assistant", "content": "x", "tool_call_id": "t1"}],
                    },
                ),
                (
                    "harness_tool_missing_tool_call_id",
                    "post",
                    _COMPLETE,
                    {"backend": _MODEL, "messages": [{"role": "tool", "content": "x"}]},
                ),
                ("jobs_item_no_command", "post", _JOBS_BATCH, {"jobs": [{"extra_args": ["a"]}]}),
                (
                    "jobs_batch_field_level_whole",
                    "post",
                    _JOBS_BATCH,
                    {"jobs": [{"command": "backtest"}, {"command": 123}]},
                ),
            ),
            client,
        )
    )
    conv = _post(client, _CONV, {})
    conv_id = str(conv.json()["id"])
    out["conv_items_empty_422"] = (
        _post(client, f"{_CONV}/{conv_id}/items", {"items": []}).status_code == 422
    )
    out["conv_items_elem_str_422"] = (
        _post(client, f"{_CONV}/{conv_id}/items", {"items": ["x"]}).status_code == 422
    )
    out["conv_item_ids_refused_422"] = (
        _post(
            client, f"{_CONV}/{conv_id}/items", {"item_ids": ["msg_x"], "items": [_msg("x")]}
        ).status_code
        == 422
    )
    out["msgb_bad_params_whole_422"] = (
        _post(
            client,
            _MSG_BATCH,
            {
                "requests": [
                    {"custom_id": "a", "params": _msg_body()},
                    {"custom_id": "b", "params": {**_msg_body(), "max_tokens": "hot"}},
                ]
            },
        ).status_code
        == 422
    )
    out["msgb_dup_custom_id_422"] = (
        _post(
            client,
            _MSG_BATCH,
            {
                "requests": [
                    {"custom_id": "a", "params": _msg_body()},
                    {"custom_id": "a", "params": _msg_body()},
                ]
            },
        ).status_code
        == 422
    )
    out["score_input_list_int_422"] = _post(client, _SCORE, {"input": ["x", 5]}).status_code == 422
    # Translation-layer structural refusals land 400, not 422.
    out["chat_content_missing_400"] = (
        _post(client, _CHAT, {"model": _MODEL, "messages": [{"role": "user"}]}).status_code == 400
    )
    out["chat_content_null_400"] = _post(client, _CHAT, _chat(None)).status_code == 400
    out["chat_tool_calls_malformed_400"] = (
        _post(
            client,
            _CHAT,
            _chat(messages=[{"role": "assistant", "content": "x", "tool_calls": [{"id": 1}]}]),
        ).status_code
        == 400
    )
    out["chat_tool_calls_wrong_role_400"] = (
        _post(
            client,
            _CHAT,
            _chat(
                messages=[
                    {
                        "role": "user",
                        "content": "x",
                        "tool_calls": [
                            {
                                "id": "t",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            ),
        ).status_code
        == 400
    )
    out["chat_tool_call_id_wrong_role_400"] = (
        _post(
            client,
            _CHAT,
            _chat(messages=[{"role": "assistant", "content": "x", "tool_call_id": "t"}]),
        ).status_code
        == 400
    )
    # Well-formed tool-calling requests validate, then hit the honest
    # capability verdict — the stub has no structured-completion channel.
    ok_chat = _post(client, _CHAT, _chat(tools=_tools(2)))
    out["chat_tools_validated_then_501"] = ok_chat.status_code == 501 and _err_code(ok_chat) in {
        "not_implemented",
        "not_supported",
    }
    ok_msg = _post(
        client, _MSG, _msg_body(tools=[{"name": "f", "input_schema": {"type": "object"}}])
    )
    out["msg_tools_validated_then_501"] = ok_msg.status_code == 501
    ok_tc = _post(
        client,
        _COMPLETE,
        {
            "backend": _MODEL,
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "y", "tool_call_id": "t1"},
                {"role": "user", "content": "x"},
            ],
        },
    )
    out["harness_tool_calls_wellformed_then_501"] = ok_tc.status_code == 501
    # Duplicate tool names are accepted on the chat grammar.
    dup = _post(
        client,
        _CHAT,
        _chat(
            tools=[
                {"type": "function", "function": {"name": "f"}},
                {"type": "function", "function": {"name": "f"}},
            ]
        ),
    )
    out["chat_tools_dup_names_validated_then_501"] = dup.status_code == 501
    return out


# ---------------------------------------------------------------------------
# Cross-field coherence
# ---------------------------------------------------------------------------


def _cross_field_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    out.update(
        _refusals(
            (
                (
                    "chat_max_tokens_conflict",
                    "post",
                    _CHAT,
                    _chat(max_tokens=8, max_completion_tokens=9),
                ),
                ("chat_top_logprobs_without_logprobs", "post", _CHAT, _chat(top_logprobs=2)),
                (
                    "chat_tool_choice_fn_without_tools",
                    "post",
                    _CHAT,
                    _chat(tool_choice={"type": "function", "function": {"name": "f"}}),
                ),
                ("chat_tool_choice_auto_without_tools", "post", _CHAT, _chat(tool_choice="auto")),
                (
                    "chat_parallel_tool_calls_without_tools",
                    "post",
                    _CHAT,
                    _chat(parallel_tool_calls=True),
                ),
                (
                    "resp_prev_and_conversation",
                    "post",
                    _RESP,
                    _resp(previous_response_id="resp_x", conversation="conv_x"),
                ),
                (
                    "resp_tool_choice_fs_without_fs_tool",
                    "post",
                    _RESP,
                    _resp(tool_choice={"type": "file_search"}),
                ),
                ("resp_top_logprobs_without_include", "post", _RESP, _resp(top_logprobs=2)),
                (
                    "msg_tc_tool_needs_name",
                    "post",
                    _MSG,
                    _msg_body(
                        tools=[{"name": "f", "input_schema": {"type": "object"}}],
                        tool_choice={"type": "tool"},
                    ),
                ),
                (
                    "jobs_cb_secret_without_url",
                    "post",
                    _JOBS,
                    {"command": "backtest", "callback_secret": "s"},
                ),
                # ``callback_secret`` needs ``callback_url`` — the
                # converse (url alone = unsigned callback) is legal.
                (
                    "batch_callback_secret_without_url",
                    "post",
                    _BATCHES,
                    {
                        "input_file_id": "file-x",
                        "endpoint": _CHAT,
                        "completion_window": "24h",
                        "callback_secret": "s",
                    },
                ),
                (
                    "espec_callback_fields_forbidden",
                    "post",
                    _EVALS,
                    _evalspec(callback_url="https://x"),
                ),
                (
                    "ft_callback_secret_without_url",
                    "post",
                    _FT,
                    {"model": "fx1", "training_file": "file-x", "callback_secret": "s"},
                ),
                (
                    "msgb_callback_secret_without_url",
                    "post",
                    _MSG_BATCH,
                    {
                        "requests": [{"custom_id": "a", "params": _msg_body()}],
                        "callback_secret": "s",
                    },
                ),
            ),
            client,
        )
    )
    # An unsigned callback is legal — ``callback_url`` alone submits fine.
    cb_file = _upload_jsonl(
        client,
        [
            json.dumps(
                {
                    "custom_id": "g",
                    "method": "POST",
                    "url": _CHAT,
                    "body": {"messages": [{"role": "user", "content": "x"}]},
                }
            )
        ],
    )
    out["batch_callback_url_unsigned_ok"] = (
        _post(
            client,
            _BATCHES,
            {
                "input_file_id": cb_file,
                "endpoint": _CHAT,
                "completion_window": "24h",
                "callback_url": "https://example.com/cb",
            },
        ).status_code
        == 200
    )
    eval_id = _eval_id(client)
    out["evalrun_source_nondict_422"] = (
        _post(
            client, f"{_EVALS}/{eval_id}/runs", {"model": _MODEL, "data_source": {"source": "nope"}}
        ).status_code
        == 422
    )
    out["evalrun_source_badmerge_422"] = (
        _post(
            client,
            f"{_EVALS}/{eval_id}/runs",
            {"model": _MODEL, "data_source": {"source": {"suite": "bogus"}}},
        ).status_code
        == 422
    )
    out["evalrun_judge_byok_wrong_backend_422"] = (
        _post(
            client,
            f"{_EVALS}/{eval_id}/runs",
            {
                "model": _MODEL,
                "judge_byok": {"base_url": "https://x", "api_key": "k", "model": "m"},
            },
        ).status_code
        == 422
    )
    out["resp_bg_nostore_400"] = (
        _post(client, _RESP, _resp(background=True, store=False)).status_code == 400
    )
    bg_stream = _post(client, _RESP, _resp(background=True, stream=True))
    out["resp_bg_stream_streams"] = (
        bg_stream.status_code == 200 and "response.created" in bg_stream.text
    )
    out["chat_max_tokens_agree_ok"] = (
        _post(client, _CHAT, _chat(max_tokens=8, max_completion_tokens=8)).status_code == 200
    )
    lp = _post(client, _CHAT, _chat(logprobs=True, top_logprobs=2))
    out["chat_logprobs_with_top_validated_501"] = lp.status_code == 501
    out["chat_stream_options_without_stream_ok"] = (
        _post(client, _CHAT, _chat(stream_options={"include_usage": True})).status_code == 200
    )
    out["chat_stream_plus_store_false_ok"] = (
        _post(client, _CHAT, _chat(stream=True, store=False)).status_code == 200
    )
    out["chat_temp_and_top_p_ok"] = (
        _post(client, _CHAT, _chat(temperature=0.7, top_p=0.9)).status_code == 200
    )
    tc_fn = _post(
        client,
        _CHAT,
        _chat(tools=_tools(1), tool_choice={"type": "function", "function": {"name": "f0"}}),
    )
    out["chat_tool_choice_fn_with_tools_validated"] = tc_fn.status_code == 501
    out["resp_reasoning_effort_ok"] = (
        _post(client, _RESP, _resp(reasoning={"effort": "low"})).status_code == 200
    )
    out["jobs_cb_url_without_secret_ok"] = _post(
        client, _JOBS, {"command": "backtest", "callback_url": "https://example.com/cb"}
    ).status_code in (200, 202)
    return out


# ---------------------------------------------------------------------------
# Metadata policies
# ---------------------------------------------------------------------------


def _metadata_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    out.update(
        _refusals(
            (
                ("chat_metadata_17", "post", _CHAT, _chat(metadata=_meta(17))),
                ("chat_metadata_key_65", "post", _CHAT, _chat(metadata={"k" * 65: "v"})),
                ("chat_metadata_value_513", "post", _CHAT, _chat(metadata={"k": "v" * 513})),
                ("chat_metadata_value_int", "post", _CHAT, _chat(metadata={"k": 1})),
                (
                    "batch_metadata_17",
                    "post",
                    _BATCHES,
                    {
                        "input_file_id": "file-x",
                        "endpoint": _CHAT,
                        "completion_window": "24h",
                        "metadata": _meta(17),
                    },
                ),
                (
                    "ft_metadata_17",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "metadata": _meta(17)},
                ),
                ("espec_metadata_value_int", "post", _EVALS, _evalspec(metadata={"k": 1})),
            ),
            client,
        )
    )
    out["chat_metadata_16_ok"] = _post(client, _CHAT, _chat(metadata=_meta(16))).status_code == 200
    out["chat_metadata_key_64_ok"] = (
        _post(client, _CHAT, _chat(metadata={"k" * 64: "v"})).status_code == 200
    )
    # Vector stores enforce the cap at route level — 400, not 422.
    out["vs_metadata_17_refused_400"] = (
        _post(client, _VS, {"name": "s", "metadata": _meta(17)}).status_code == 400
    )
    out["vs_metadata_16_ok"] = (
        _post(client, _VS, {"name": "s", "metadata": _meta(16)}).status_code == 200
    )
    # Honest inconsistency pin: eval-spec metadata declares no bound —
    # 17 keys are accepted where chat/batch/vs all refuse. The battery
    # records the gap rather than laundering it.
    out["espec_metadata_unbounded_17_accepted"] = (
        _post(client, _EVALS, _evalspec(metadata=_meta(17))).status_code == 201
    )
    out["conv_metadata_ok"] = _post(client, _CONV, {"metadata": {"k": "v"}}).status_code == 200
    return out


# ---------------------------------------------------------------------------
# Extra-field binding policy — allow swallows, forbid refuses
# ---------------------------------------------------------------------------


def _extra_fields_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    for name, path, body in (
        ("chat_extra_swallowed", _CHAT, _chat(bogus_field=1)),
        (
            "chat_msg_extra_swallowed",
            _CHAT,
            _chat(messages=[{"role": "user", "content": "x", "bogus": 1}]),
        ),
        ("resp_extra_swallowed", _RESP, _resp(bogus_field=1)),
        ("msg_extra_swallowed", _MSG, _msg_body(bogus_field=1)),
        ("legacy_extra_swallowed", _LEGACY, _legacy(bogus_field=1)),
        ("emb_extra_swallowed", _EMB, _emb(bogus_field=1)),
        ("conv_extra_swallowed", _CONV, {"bogus_field": 1}),
        ("vs_extra_swallowed", _VS, {"name": "s", "bogus_field": 1}),
        (
            "msgb_extra_swallowed",
            _MSG_BATCH,
            {"requests": [{"custom_id": "a", "params": _msg_body()}], "bogus_field": 1},
        ),
    ):
        out[f"{name}_ok"] = _post(client, path, body).status_code in (200, 201)
    out.update(
        _refusals(
            (
                ("harness_extra_forbidden", "post", _COMPLETE, _complete(bogus_field=1)),
                (
                    "harness_served_model_forge",
                    "post",
                    _COMPLETE,
                    _complete(**{"_served_model": "forged"}),
                ),
                ("fx1_extra_forbidden", "post", _CHAT, _chat(fx1={"bogus": 1})),
                (
                    "cb_extra_forbidden",
                    "post",
                    _COMPLETE_BATCH,
                    {"backend": _MODEL, "batch": [_msgs(1)], "bogus_field": 1},
                ),
                ("key_extra_forbidden", "post", _KEYS, {"bogus_field": 1}),
                ("espec_extra_forbidden", "post", _EVALS, _evalspec(bogus_field=1)),
                (
                    "upload_extra_forbidden",
                    "post",
                    _UPLOADS,
                    {
                        "purpose": "batch",
                        "filename": "f.jsonl",
                        "bytes": 4,
                        "mime_type": "t",
                        "bogus_field": 1,
                    },
                ),
                (
                    "ft_extra_forbidden",
                    "post",
                    _FT,
                    {"model": "m", "training_file": "file-x", "bogus_field": 1},
                ),
                ("jobs_extra_forbidden", "post", _JOBS, {"command": "backtest", "bogus_field": 1}),
                (
                    "jobs_batch_extra_forbidden",
                    "post",
                    _JOBS_BATCH,
                    {"jobs": [{"command": "backtest"}], "bogus_field": 1},
                ),
                (
                    "evalsubmit_extra_forbidden",
                    "post",
                    _EVALS_H,
                    {"suite": "capability", "backend": _MODEL, "bogus_field": 1},
                ),
                (
                    "verify_extra_forbidden",
                    "post",
                    _VERIFY,
                    {"receipt": {"x": 1}, "bogus_field": 1},
                ),
                ("probe_extra_forbidden", "post", _PROBE, {"bogus": 1}),
                ("gate_extra_forbidden", "post", _GATE, {"text": "x", "bogus_field": 1}),
                ("score_extra_forbidden", "post", _SCORE, {"input": "x", "bogus_field": 1}),
                ("moderation_extra_forbidden", "post", _MOD, {"input": "x", "bogus_field": 1}),
            ),
            client,
        )
    )
    eval_id = _eval_id(client)
    out["evalrun_extra_forbidden"] = (
        _post(client, f"{_EVALS}/{eval_id}/runs", {"model": _MODEL, "bogus_field": 1}).status_code
        == 422
    )
    # Patch-model nullable fields: scopes/admin sent as null refuse —
    # clearing is a separate, explicit operation.
    _key, kid = _mint(client)
    out["key_patch_scopes_null_422"] = (
        _req(client, "patch", f"{_KEYS}/{kid}", body={"scopes": None}).status_code == 422
    )
    out["key_patch_admin_null_422"] = (
        _req(client, "patch", f"{_KEYS}/{kid}", body={"admin": None}).status_code == 422
    )
    out["key_patch_extra_forbidden_422"] = (
        _req(client, "patch", f"{_KEYS}/{kid}", body={"bogus": 1}).status_code == 422
    )
    return out


# ---------------------------------------------------------------------------
# Content edges, depth, and the 1MiB body cap
# ---------------------------------------------------------------------------


def _content_depth_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}

    def _deep(n: int) -> dict[str, Any]:
        d: dict[str, Any] = {"leaf": 1}
        for _ in range(n):
            d = {"k": d}
        return d

    out["chat_content_unicode_ok"] = (
        _post(client, _CHAT, _chat("say χψΩ — “unicode”")).status_code == 200
    )
    out["chat_content_ctrl_char_ok"] = _post(client, _CHAT, _chat("a\x01b")).status_code == 200
    out["chat_content_nul_ok"] = _post(client, _CHAT, _chat("a\x00b")).status_code == 200
    out["chat_content_ws_only_ok"] = _post(client, _CHAT, _chat("   \t\n  ")).status_code == 200
    out["chat_msg_extra_100deep_ok"] = (
        _post(
            client, _CHAT, _chat(messages=[{"role": "user", "content": "x", "extra": _deep(100)}])
        ).status_code
        == 200
    )
    out["chat_content_part_100deep_ok"] = (
        _post(
            client,
            _CHAT,
            _chat(
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "x", "meta": _deep(100)}]}
                ]
            ),
        ).status_code
        == 200
    )
    out["resp_input_item_100deep_ok"] = (
        _post(
            client, _RESP, _resp([{"role": "user", "content": "x", "meta": _deep(100)}])
        ).status_code
        == 200
    )
    out["msg_content_nul_ok"] = _post(client, _MSG, _msg_body("a\x00b")).status_code == 200
    out["gate_text_nul_ok"] = _post(client, _GATE, {"text": "a\x00b"}).status_code == 200
    out["legacy_prompt_nul_ok"] = (
        _post(client, _LEGACY, _legacy(prompt="a\x00b")).status_code == 200
    )
    # The 1MiB ingress cap is the real size boundary: just under lands,
    # over refuses 413 too_large.
    under = _post(client, _CHAT, _chat("x" * (900 << 10)))
    out["chat_body_just_under_1mb_ok"] = under.status_code == 200
    over = _post(client, _CHAT, _chat("x" * ((1 << 20) + 64)))
    out["chat_body_over_1mb_413"] = over.status_code == 413 and _err_code(over) == "too_large"
    resp_over = _post(client, _RESP, _resp("x" * ((1 << 20) + 64)))
    out["resp_body_over_1mb_413"] = resp_over.status_code == 413
    return out


# ---------------------------------------------------------------------------
# Batch-line granularity — per-line errored rows vs whole-submit refusals
# ---------------------------------------------------------------------------


def _upload_jsonl(client: TestClient, lines: list[str], *, name: str = "b.jsonl") -> str:
    blob = "\n".join(lines).encode() + b"\n"
    r = client.post(
        _FILES,
        files={"file": (name, io.BytesIO(blob))},
        data={"purpose": "batch"},
        headers=_H,
    )
    _note(_FILES, r)
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _batch_wait(client: TestClient, batch_id: str) -> dict[str, Any]:
    deadline: dict[str, Any] = {}

    def _done() -> bool:
        r = _get(client, f"{_BATCHES}/{batch_id}")
        if r.status_code == 200 and r.json().get("status") in (
            "completed",
            "failed",
            "expired",
            "cancelled",
        ):
            deadline["r"] = dict(r.json())
            return True
        return False

    assert _wait_for(_done, timeout_s=60.0), f"batch {batch_id} never settled"
    return cast("dict[str, Any]", deadline["r"])


def _batch_line_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    good = {"custom_id": "g", "method": "POST", "url": _CHAT, "body": _chat()}
    bad_field = {"custom_id": "b", "method": "POST", "url": _CHAT, "body": _chat(temperature="hot")}
    stream_line = {"custom_id": "s", "method": "POST", "url": _CHAT, "body": _chat(stream=True)}
    bg_line = {
        "custom_id": "bg",
        "method": "POST",
        "url": _CHAT,
        "body": {**_chat(), "store": False, "background": True},
    }
    conv_line = {
        "custom_id": "c",
        "method": "POST",
        "url": _CHAT,
        "body": {**_chat(), "conversation": "conv_nope"},
    }
    file_id = _upload_jsonl(
        client,
        [json.dumps(ln) for ln in (good, bad_field, stream_line, bg_line, conv_line)],
    )
    r = _post(
        client, _BATCHES, {"input_file_id": file_id, "endpoint": _CHAT, "completion_window": "24h"}
    )
    batch = _batch_wait(client, str(r.json()["id"])) if r.status_code == 200 else {}
    counts = batch.get("request_counts", {})
    out["batch_submit_ok"] = r.status_code == 200
    out["batch_completes_with_errored_lines"] = batch.get("status") == "completed"
    out["batch_request_counts_honest"] = (
        counts.get("total") == 5 and counts.get("completed") == 1 and counts.get("failed") == 4
    )
    out_id = batch.get("output_file_id")
    rows = []
    if out_id:
        content = _get(client, f"{_FILES}/{out_id}/content")
        if content.status_code == 200:
            rows = [json.loads(ln) for ln in content.text.splitlines() if ln.strip()]
    by_id = {row.get("custom_id"): row for row in rows}
    resp_codes = {cid: (row.get("response") or {}).get("status_code") for cid, row in by_id.items()}
    out["batch_good_line_200_row"] = resp_codes.get("g") == 200
    out["batch_field_violation_errored_row"] = resp_codes.get("b") == 400
    out["batch_stream_line_errored_row"] = resp_codes.get("s") in (400, 422)
    out["batch_background_line_errored_row"] = resp_codes.get("bg") in (400, 422)
    out["batch_conversation_line_errored_row"] = resp_codes.get("c") in (400, 422)
    out["batch_no_row_is_5xx"] = bool(rows) and all(
        isinstance(v, int) and v < 500 for v in resp_codes.values()
    )
    # Shape-level violations refuse the whole submit before any line runs.
    bad_shape_file = _upload_jsonl(
        client, [json.dumps({"method": "POST", "url": _CHAT, "body": _chat()})]
    )
    out["batch_shape_error_whole_400"] = (
        _post(
            client,
            _BATCHES,
            {"input_file_id": bad_shape_file, "endpoint": _CHAT, "completion_window": "24h"},
        ).status_code
        == 400
    )
    wrong_ep_file = _upload_jsonl(
        client, [json.dumps({"custom_id": "w", "method": "POST", "url": _EMB, "body": _emb()})]
    )
    out["batch_wrong_endpoint_whole_400"] = (
        _post(
            client,
            _BATCHES,
            {"input_file_id": wrong_ep_file, "endpoint": _CHAT, "completion_window": "24h"},
        ).status_code
        == 400
    )
    get_req_file = _upload_jsonl(
        client, [json.dumps({"custom_id": "w", "method": "GET", "url": _CHAT, "body": _chat()})]
    )
    out["batch_get_method_whole_400"] = (
        _post(
            client,
            _BATCHES,
            {"input_file_id": get_req_file, "endpoint": _CHAT, "completion_window": "24h"},
        ).status_code
        == 400
    )
    return out


# ---------------------------------------------------------------------------
# Surface edges — per-surface happy paths + capability verdicts
# ---------------------------------------------------------------------------


def _surface_edge_probes() -> dict[str, bool]:
    stub = _FullBackend()
    client, _app = _client(_backends(stub), api_key=_ROOT)
    out: dict[str, bool] = {}
    # Validated happy paths through the optional channels.
    emb = _post(client, _EMB, _emb())
    out["emb_validated_200"] = emb.status_code == 200
    ct = _post(client, _MSG_COUNT, _msg_body())
    out["count_tokens_validated_200"] = ct.status_code == 200 and ct.json().get("input_tokens") == 7
    out["count_tokens_tools_refused_400"] = (
        _post(
            client, _MSG_COUNT, _msg_body(tools=[{"name": "f", "input_schema": {"type": "object"}}])
        ).status_code
        == 400
    )
    score = _post(client, _SCORE, {"input": "x"})
    out["score_validated_200"] = score.status_code == 200
    gate = _post(client, _GATE, {"text": "the model said hi"})
    out["gate_validated_200"] = gate.status_code == 200 and "ok" in gate.json()
    mod = _post(client, _MOD, {"input": "hello"})
    out["moderation_validated_200"] = mod.status_code == 200
    probe = _post(client, _PROBE, {})
    out["backend_probe_validated_200"] = probe.status_code == 200 and probe.json().get("ok") is True
    evalsub = _post(client, _EVALS_H, {"suite": "capability", "backend": _MODEL})
    out["evalsubmit_validated_202"] = evalsub.status_code == 202
    eval_id = _eval_id(client)
    run = _post(client, f"{_EVALS}/{eval_id}/runs", {"model": _MODEL})
    out["evalrun_validated_201"] = run.status_code == 201
    conv = _post(client, _CONV, {})
    out["conv_create_200"] = conv.status_code == 200
    conv_id = str(conv.json()["id"])
    items = _post(client, f"{_CONV}/{conv_id}/items", {"items": [_msg("hi")]})
    out["conv_items_add_2xx"] = items.status_code in (200, 201)
    vs = _post(client, _VS, {"name": "s"})
    out["vs_create_200"] = vs.status_code == 200
    vs_id = str(vs.json()["id"])
    file_id = _upload_jsonl(client, ['{"a": 1}'])
    vf = _post(client, f"{_VS}/{vs_id}/files", {"file_id": file_id})
    out["vs_file_create_ok"] = vf.status_code in (200, 201)
    out["vs_file_empty_id_422"] = (
        _post(client, f"{_VS}/{vs_id}/files", {"file_id": ""}).status_code == 422
    )
    search = _post(client, f"{_VS}/{vs_id}/search", {"query": "q"})
    out["vs_search_query_ok"] = search.status_code == 200
    out["vs_search_query_empty_422"] = (
        _post(client, f"{_VS}/{vs_id}/search", {"query": []}).status_code == 422
    )
    out["vs_search_query_dict_422"] = (
        _post(client, f"{_VS}/{vs_id}/search", {"query": {}}).status_code == 422
    )
    # Upload lifecycle: create → parts → md5-checked complete mints a file.
    up = _post(
        client,
        _UPLOADS,
        {"purpose": "batch", "filename": "up.jsonl", "bytes": 4, "mime_type": "text/jsonl"},
    )
    up_id = str(up.json()["id"])
    p1 = _req(
        client, "post", f"{_UPLOADS}/{up_id}/parts", files={"data": ("p1", io.BytesIO(b"ab"))}
    )
    p2 = _req(
        client, "post", f"{_UPLOADS}/{up_id}/parts", files={"data": ("p2", io.BytesIO(b"cd"))}
    )
    ids = [str(p1.json()["id"]), str(p2.json()["id"])]
    done = _post(
        client,
        f"{_UPLOADS}/{up_id}/complete",
        {"part_ids": ids, "md5": hashlib.md5(b"abcd", usedforsecurity=False).hexdigest()},
    )
    out["upload_lifecycle_200"] = (
        up.status_code == 200
        and p1.status_code == 200
        and p2.status_code == 200
        and done.status_code == 200
    )
    out["upload_md5_short_422"] = (
        _post(
            client, f"{_UPLOADS}/{up_id}/complete", {"part_ids": ids, "md5": "x" * 31}
        ).status_code
        == 422
    )
    out["upload_part_ids_over_422"] = (
        _post(
            client,
            f"{_UPLOADS}/{up_id}/complete",
            {"part_ids": [f"part_{i}" for i in range(65)], "md5": "x" * 32},
        ).status_code
        == 422
    )
    # Verdict endpoints on malformed payloads.
    nonreceipt = _post(client, _VERIFY, {"receipt": {"x": 1}})
    out["verify_nonreceipt_valid_false"] = (
        nonreceipt.status_code == 200 and nonreceipt.json().get("valid") is False
    )
    committed = Path(__file__).resolve().parents[3] / "receipts" / "api_audit.json"
    if committed.exists():
        payload = json.loads(committed.read_text())
        vr = _post(client, _VERIFY, {"receipt": payload})
        out["verify_committed_receipt_valid_true"] = (
            vr.status_code == 200 and vr.json().get("valid") is True
        )
    # Files: upload then retrieve/delete roundtrip.
    fid = _upload_jsonl(client, ['{"a": 1}'])
    out["files_retrieve_200"] = _get(client, f"{_FILES}/{fid}").status_code == 200
    out["files_purpose_bogus_400"] = (
        _req(
            client,
            "post",
            _FILES,
            files={"file": ("f.jsonl", io.BytesIO(b'{"a":1}\n'))},
            data={"purpose": "bogus"},
        ).status_code
        == 400
    )
    out["files_empty_400"] = (
        _req(
            client,
            "post",
            _FILES,
            files={"file": ("f.jsonl", io.BytesIO(b""))},
            data={"purpose": "batch"},
        ).status_code
        == 400
    )
    out["files_nonjsonl_400"] = (
        _req(
            client,
            "post",
            _FILES,
            files={"file": ("f.txt", io.BytesIO(b"hi"))},
            data={"purpose": "batch"},
        ).status_code
        == 400
    )
    out["files_missing_field_400"] = (
        _req(client, "post", _FILES, data={"purpose": "batch"}).status_code == 400
    )
    # Jobs + anthropic batch submits (async verdicts).
    job = _post(client, _JOBS, {"command": "backtest"})
    out["jobs_submit_2xx"] = job.status_code in (200, 202)
    msgb = _post(client, _MSG_BATCH, {"requests": [{"custom_id": "a", "params": _msg_body()}]})
    out["msgb_submit_2xx"] = msgb.status_code in (200, 201, 202)
    # Non-JSON and truncated bodies refuse at parse — never a 500.
    out["body_nonjson_422"] = _req(client, "post", _CHAT, raw=b"not json{").status_code == 422
    out["body_truncated_422"] = _req(client, "post", _CHAT, raw=b'{"messages":').status_code == 422
    out["body_json_array_422"] = _req(client, "post", _COMPLETE, raw=b"[]").status_code == 422
    return out


# ---------------------------------------------------------------------------
# Error shape — every refusal lands the declared envelope, never a 500
# ---------------------------------------------------------------------------


def _error_shape_probes() -> dict[str, bool]:
    client, _app = _client(_backends(), api_key=_ROOT)
    out: dict[str, bool] = {}
    h = _post(client, _COMPLETE, _complete(temperature="hot"))
    hbody = h.json()
    out["err_harness_422_detail_list"] = (
        h.status_code == 422
        and isinstance(hbody.get("detail"), list)
        and hbody.get("code") == "validation"
    )
    v = _post(client, _CHAT, _chat(temperature="hot"))
    vbody = v.json()
    verr = vbody.get("error") or {}
    out["err_openai_422_envelope"] = (
        v.status_code == 422
        and verr.get("code") == "validation"
        and isinstance(verr.get("message"), str)
        and verr.get("type") == "invalid_request_error"
    )
    a = _post(client, _MSG, _msg_body(max_tokens=0))
    abody = a.json()
    out["err_anthropic_422_envelope"] = (
        a.status_code == 422
        and abody.get("type") == "error"
        and isinstance(abody.get("error"), dict)
        and abody["error"].get("type") == "invalid_request_error"
    )
    t = _post(
        client,
        _CHAT,
        _chat(
            messages=[
                {
                    "role": "user",
                    "content": "x",
                    "tool_calls": [
                        {
                            "id": "t",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ]
        ),
    )
    out["err_translation_400_envelope"] = t.status_code == 400 and _err_code(t) == "invalid_request"
    cap = _post(client, _CHAT, _chat(tools=_tools(1)))
    out["err_capability_501_envelope"] = (
        cap.status_code == 501
        and _err_code(cap) in {"not_implemented", "not_supported"}
        and isinstance(cap.json().get("error"), dict)
    )
    nf = _get(client, f"{_JOBS}/job_nonexistent")
    out["err_harness_route_404_envelope"] = (
        nf.status_code == 404 and nf.json().get("code") == "not_found"
    )
    out["err_no_bare_500"] = all(status != 500 for _p, status, _c in _SEEN)
    out["err_every_5xx_declared"] = all(
        status < 500 or (code is not None and code in _HONEST_5XX) for _p, status, code in _SEEN
    )
    return out


# ---------------------------------------------------------------------------
# Validity invariant — refused work burns nothing, stores nothing
# ---------------------------------------------------------------------------


def _validity_invariant_probes() -> dict[str, bool]:
    stub = _FullBackend()
    client, _app = _client(_backends(stub), api_key=_ROOT)
    out: dict[str, bool] = {}
    key, _kid = _mint(client, name="tracked")
    kh = {"X-API-Key": key}

    def _self() -> dict[str, Any]:
        return _get(client, _SELF, headers=kh).json()["key"]  # type: ignore[no-any-return]

    refused = (
        (_CHAT, _chat(temperature="hot")),
        (_CHAT, _chat(n=0)),
        (_CHAT, _chat(messages=[])),
        (_RESP, _resp(max_output_tokens=0)),
        (_MSG, _msg_body(max_tokens=0)),
        (_LEGACY, _legacy(n=9)),
        (_EMB, _emb(input={})),
        (_COMPLETE, _complete(temperature=2.1)),
        (_JOBS, {"command": 123}),
    )
    calls0 = int(stub.calls)
    u0 = int(_self()["uses"])
    served0 = int(_self()["served"]["calls"])
    listed0 = _get(client, f"{_CHAT}?limit=100", headers=kh)
    n0 = len(listed0.json().get("data", [])) if listed0.status_code == 200 else -1
    for path, body in refused:
        _post(client, path, body, headers=kh)
    out["inv_refusal_burns_no_backend"] = int(stub.calls) == calls0
    u1 = int(_self()["uses"])
    served1 = int(_self()["served"]["calls"])
    out["inv_refusal_no_served_calls"] = served1 == served0
    # Refusals land AFTER auth — the credential honestly bills the
    # attempt (each refusal is still a served request for rate limits).
    out["inv_refusal_bills_uses"] = u1 - u0 >= len(refused)
    listed1 = _get(client, f"{_CHAT}?limit=100", headers=kh)
    n1 = len(listed1.json().get("data", [])) if listed1.status_code == 200 else -1
    out["inv_refusal_stores_no_record"] = n0 >= 0 and n1 == n0
    # Contrast: a well-formed call stores exactly one record and bills.
    ok = _post(client, _CHAT, _chat("stored"), headers=kh)
    out["inv_ok_call_200"] = ok.status_code == 200
    listed2 = _get(client, f"{_CHAT}?limit=100", headers=kh)
    n2 = len(listed2.json().get("data", [])) if listed2.status_code == 200 else -1
    out["inv_ok_call_stores_record"] = n2 == n1 + 1
    rid = str(ok.json()["id"])
    out["inv_ok_call_retrievable_200"] = (
        _get(client, f"{_CHAT}/{rid}", headers=kh).status_code == 200
    )
    # store:false is the honest skip — the record mints, serves, and is
    # then unreachable (404 evicted/never-stored verdict).
    ns = _post(client, _CHAT, _chat("nostore", store=False), headers=kh)
    out["inv_store_false_200"] = ns.status_code == 200
    out["inv_store_false_404"] = (
        _get(client, f"{_CHAT}/{ns.json()['id']}", headers=kh).status_code == 404
    )
    rok = _post(client, _RESP, _resp("stored"), headers=kh)
    out["inv_ok_response_200"] = rok.status_code == 200
    out["inv_ok_response_retrievable"] = (
        _get(client, f"{_RESP}/{rok.json()['id']}", headers=kh).status_code == 200
    )
    rns = _post(client, _RESP, _resp("nostore", store=False), headers=kh)
    out["inv_resp_store_false_404"] = (
        _get(client, f"{_RESP}/{rns.json()['id']}", headers=kh).status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Battery + sealed receipt
# ---------------------------------------------------------------------------


def schema_audit() -> dict[str, bool]:
    """Run the validation-edge battery; returns literal bools."""
    _SEEN.clear()
    with _audit_context():
        out: dict[str, bool] = {}
        out.update(_type_strictness_probes())
        out.update(_bounds_probes())
        out.update(_enum_literal_probes())
        out.update(_structure_probes())
        out.update(_cross_field_probes())
        out.update(_metadata_probes())
        out.update(_extra_fields_probes())
        out.update(_content_depth_probes())
        out.update(_batch_line_probes())
        out.update(_surface_edge_probes())
        out.update(_validity_invariant_probes())
        out.update(_error_shape_probes())  # last — reads the whole ledger
        return out


def schema_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under schema_audit.v1."""
    r = schema_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "schema_audit",
        "schema": "schema_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends",
            "not_executed": [
                "provider-side validation beyond this grammar",
                "live network bind, TLS, or disconnect timing",
                "auth/signature rotation edges (auth_audit covers)",
                "response-body validation beyond declared verdicts",
            ],
        },
        "interpretation": (
            "Every request model's declared edges are real boundaries, "
            "measured end to end: the coercion policy is pydantic lax — "
            "parseable scalars coerce ('0.7'→float, '2'→int, 'yes'→SSE "
            "stream, True→1) while wrong types refuse 422; every "
            "declared bound refuses past its cap and accepts at it "
            "(n caps at 8 on this harness, the body cap is 1MiB → 413); "
            "enum fields refuse out-of-domain values and role strictness "
            "is surface-dependent (chat/harness free-form accepted, "
            "responses refused at translation, Anthropic Literal 422); "
            "cross-field rules hold (max_tokens agreement, "
            "top_logprobs↔logprobs, tool_choice↔tools, "
            "previous_response_id⊥conversation, background⊃store, "
            "stream≻background documented precedence); metadata pins "
            "16/64/512 on chat and batch, 400 at route level on vector "
            "stores, and is honestly unbounded on eval specs (pinned, "
            "not laundered); extra=allow models swallow unknown keys "
            "while extra=forbid models refuse including forged private "
            "attrs; control chars, NUL, unicode, and whitespace content "
            "execute verbatim; batch field violations land per-line "
            "errored rows while shape violations refuse the whole "
            "submit; every refusal carries the declared envelope (harness "
            "{detail,code:validation} or the OpenAI/Anthropic error "
            "envelope) and no probe anywhere observed a bare 500 or an "
            "undeclared 5xx; refused work burns no backend call, stores "
            "no record, and logs no served-call while uses honestly "
            "bills the auth'd attempt; store:false mints then 404s."
            if ok
            else f"SCHEMA AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(schema_audit_bench(), indent=2, sort_keys=True))
