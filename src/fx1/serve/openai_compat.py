"""OpenAI-compatible ingress — the shared translation layer.

The ``/v1`` HTTP routes (``fx1.serve.api``) and the in-process SDK surface
(``fx1.sdk.Fx1Harness.openai_chat``) translate through THIS module — one
pydantic validation object, one backend-resolution order, one error
taxonomy — so the wire and the weights-direct path cannot drift apart.

- *Fail-closed surface* — every OpenAI feature the gated pipeline cannot
  honor (tools, ``logprobs``, non-text ``response_format`` types, content
  parts other than ``text``…) is rejected by
  :class:`OpenAICompatError`, never silently dropped. The
  ``/v1/responses`` surface applies the same rule; ``store`` governs the
  retrieval index (``store=false`` keeps the call out of
  ``GET /v1/responses/{id}`` — the audit ledger still records it).
- *Backend precedence* — ``fx1.backend`` > ``X-Fx1-Backend`` >
  ``model`` naming a backend > ``byok`` when BYOK headers are present >
  ``hosted_k3``. Header-based BYOK treats a non-backend ``model`` as the
  upstream model name (``gpt-4o`` → base_url + key + model).
- *Error mapping* — :func:`openai_to_kwargs` raises
  :class:`OpenAICompatError` (a ``ValueError`` carrying ``status``); the
  API converts it to an :class:`ApiError` with that status, the SDK
  surfaces it as a ``ValueError`` — same verdict, each surface's own
  exception class.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "OPENAI_BACKENDS",
    "OPENAI_ERR_TYPES",
    "OPENAI_UNSUPPORTED",
    "OPENAI_MODEL_IDS",
    "OPENAI_RESPONSE_FORMATS",
    "RESPONSE_ITEM_TYPES_REFUSED",
    "RESPONSE_PART_TYPES",
    "RESPONSE_ROLES",
    "RESPONSES_UNSUPPORTED",
    "ByokOverride",
    "OpenAIChatMessage",
    "OpenAIChatResponse",
    "OpenAIChatChoice",
    "OpenAICompatError",
    "OpenAIFx1",
    "OpenAIModel",
    "OpenAIModelList",
    "OpenAIResponseRequest",
    "OPENAI_BATCH_ENDPOINTS",
    "OPENAI_BATCH_LINE_MAX",
    "OPENAI_FILE_BYTES_MAX",
    "OPENAI_FILE_PURPOSE_ACCEPT",
    "OpenAIBatchRequest",
    "batch_line_body",
    "batch_line_shape",
    "batch_object",
    "batch_output_line",
    "file_object",
    "is_openai_path",
    "openai_chunks",
    "openai_envelope",
    "openai_error_body",
    "openai_messages",
    "openai_model",
    "openai_models",
    "openai_response_events",
    "openai_response_object",
    "openai_to_kwargs",
    "openai_usage",
    "OpenAIEnvelopeStore",
    "response_input_to_messages",
    "response_text_format",
    "response_to_kwargs",
    "validate_openai_output",
    "validate_response_format",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenAICompatError(ValueError):
    """A rejection at the OpenAI translation layer.

    ``status`` is the HTTP status the wire maps the rejection to; on the
    SDK surface the exception itself is the verdict (a ``ValueError``, the
    SDK's request-error class). ``code`` lands in the OpenAI error
    envelope's ``code`` field when the wire emits it.
    """

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


# Backend names an OpenAI `model` field or `X-Fx1-Backend` header may carry.
OPENAI_BACKENDS = frozenset({"hosted_k3", "local_fx1", "byok"})

# response_format types the gated pipeline honors. `text` is freeform;
# `json_object`/`json_schema` are post-validated against the returned text
# (the harness can't constrain-decode arbitrary backends — validation is
# the honest mechanism, and a violation is a provider-side 502).
OPENAI_RESPONSE_FORMATS = frozenset({"text", "json_object", "json_schema"})

# The GET /v1/models inventory — `fx1` is the alias for the default link.
OPENAI_MODEL_IDS = ("fx1", "hosted_k3", "local_fx1", "byok")

# Fields a caller may send that the gated pipeline cannot honor. Naming the
# field beats silently dropping it — honest compat over fake compat.
# ``store`` is honored, not refused: it governs the retrieval index behind
# ``GET /v1/chat/completions/{id}`` (the audit ledger records every call
# regardless — retrieval is a convenience surface, not the evidence).
OPENAI_UNSUPPORTED = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
    "suffix",
    "echo",
    "best_of",
)

# OpenAI's `type` names per status — the error envelope stays SDK-faithful.
OPENAI_ERR_TYPES = {
    400: "invalid_request_error",
    404: "invalid_request_error",
    409: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    429: "rate_limit_error",
    503: "service_unavailable",
}


def openai_error_body(message: str, status: int, code: str) -> dict[str, Any]:
    """The OpenAI error envelope — `{error: {message, type, param, code}}`.

    ``code`` carries the harness's own code (``honesty_gate``,
    ``over_capacity``, ``draining``, …) so clients can branch on it.
    """
    return {
        "error": {
            "message": message,
            "type": OPENAI_ERR_TYPES.get(
                status, "server_error" if status >= 500 else "invalid_request_error"
            ),
            "param": None,
            "code": code,
        }
    }


def is_openai_path(path: str) -> bool:
    """Whether a request path is under the OpenAI surface."""
    return path.startswith("/v1")


class ByokOverride(_Model):
    """Per-request BYOK credentials — the caller's own OpenAI-compatible
    endpoint. ``api_key`` is used for the upstream call only; it is never
    logged and never echoed into error text. The idempotency fingerprint
    is hashed, so the stored dedupe record carries no secret bytes."""

    base_url: str = Field(min_length=1, max_length=2048)
    api_key: str = Field(min_length=1, max_length=4096)
    model: str = Field(min_length=1, max_length=512)

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        parsed = urllib.parse.urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"byok.base_url must be an http(s) URL, got {v!r}")
        return v


class OpenAIChatMessage(_Model):
    """One chat message — content may be a string or an OpenAI
    content-part list; non-text parts are rejected at translation."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class OpenAIFx1(_Model):
    """The ``fx1`` extension object: harness knobs that have no OpenAI
    field — backend selection, fallbacks, BYOK credentials, the local
    checkpoint dir, receipt citations, and the per-call deadline."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["hosted_k3", "local_fx1", "byok"] | None = None
    fallbacks: list[Literal["hosted_k3", "local_fx1", "byok"]] = Field(
        default_factory=list, max_length=2
    )
    byok: ByokOverride | None = None
    checkpoint_dir: str | None = None
    receipt_hashes: list[str] | None = None
    timeout_s: float | None = Field(default=None, gt=0, le=3600)


class OpenAIChatRequest(_Model):
    """POST /v1/chat/completions body — the OpenAI surface, extra fields
    tolerated (SDKs send bookkeeping keys like ``user``)."""

    model_config = ConfigDict(extra="allow")

    model: str = "fx1"
    messages: list[OpenAIChatMessage] = Field(min_length=1, max_length=512)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, le=262144)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=262144)
    seed: int | None = Field(default=None, ge=0)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int = Field(default=1, ge=1, le=8)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: dict[str, int] | None = None
    user: str | None = Field(default=None, max_length=512)
    metadata: dict[str, str] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    prompt_cache_key: str | None = Field(default=None, max_length=128)
    response_format: dict[str, Any] | None = None
    store: bool | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _openai_valid(self) -> OpenAIChatRequest:
        if isinstance(self.stop, list):
            if len(self.stop) > 4:
                raise ValueError("stop accepts at most 4 sequences")
            if any(not isinstance(s, str) or not 1 <= len(s) <= 512 for s in self.stop):
                raise ValueError("stop sequences must be 1–512 char strings")
        elif isinstance(self.stop, str) and not 1 <= len(self.stop) <= 512:
            raise ValueError("stop sequences must be 1–512 char strings")
        if self.logit_bias is not None:
            for key, bias in self.logit_bias.items():
                try:
                    int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"logit_bias keys must be token ids, got {key!r}") from exc
                if not -100 <= bias <= 100:
                    raise ValueError(f"logit_bias[{key!r}]={bias} outside [-100, 100]")
        if (
            self.max_completion_tokens is not None
            and self.max_tokens is not None
            and self.max_completion_tokens != self.max_tokens
        ):
            raise ValueError(
                "max_tokens and max_completion_tokens disagree "
                f"({self.max_tokens} vs {self.max_completion_tokens})"
            )
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        rf = self.response_format
        if rf is not None:
            rtype = rf.get("type", "text")
            if rtype not in OPENAI_RESPONSE_FORMATS:
                raise ValueError(
                    "response_format.type must be one of "
                    f"{sorted(OPENAI_RESPONSE_FORMATS)}; got {rtype!r}"
                )
            if rtype == "json_schema":
                spec = rf.get("json_schema")
                schema = spec.get("schema") if isinstance(spec, dict) else None
                if not isinstance(schema, dict):
                    raise ValueError(
                        "response_format json_schema needs "
                        "{type: 'json_schema', json_schema: {name?, schema: {...}}}"
                    )
                try:
                    jsonschema.validators.validator_for(schema).check_schema(schema)
                except jsonschema.SchemaError as exc:
                    raise ValueError(
                        f"response_format json_schema is not a valid schema: {exc.message}"
                    ) from exc
        present = [f for f in OPENAI_UNSUPPORTED if getattr(self, f, None) is not None]
        # extras (extra="allow") that are also unsupported features
        extra_bad = sorted(f for f in OPENAI_UNSUPPORTED if f in (self.__pydantic_extra__ or {}))
        bad = sorted(set(present) | set(extra_bad))
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


class OpenAIModel(_Model):
    """One entry of GET /v1/models — `id` is the name `model` may carry."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "dipcatcher"


class OpenAIModelList(_Model):
    """GET /v1/models body — the OpenAI `list` envelope."""

    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class OpenAIChatChoice(_Model):
    """One choice of a `chat.completion` — the gated text lands here."""

    index: int
    message: dict[str, str]
    finish_reason: Literal["stop"]


class OpenAIChatResponse(_Model):
    """POST /v1/chat/completions 200 body — the `chat.completion`
    envelope. `system_fingerprint` carries the serving backend name;
    `usage` is null when the provider didn't report token counts."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: list[OpenAIChatChoice]
    usage: dict[str, int] | None = None


def openai_models(*, created: int | None = None) -> OpenAIModelList:
    """The model inventory — `fx1` plus the backend names `model` may
    carry. ``created`` defaults to call time."""
    ts = int(time.time()) if created is None else created
    return OpenAIModelList(data=[OpenAIModel(id=m, created=ts) for m in OPENAI_MODEL_IDS])


def openai_model(model_id: str, *, created: int | None = None) -> OpenAIModel:
    """One model card — ``GET /v1/models/{id}`` retrieve semantics.

    Unknown ids fail closed 404 (OpenAI's ``invalid_request_error`` /
    ``model_not_found``) — an SDK's ``models.retrieve`` never gets a
    fabricated card.
    """
    if model_id not in OPENAI_MODEL_IDS:
        raise OpenAICompatError(
            f"The model '{model_id}' does not exist", status=404, code="model_not_found"
        )
    ts = int(time.time()) if created is None else created
    return OpenAIModel(id=model_id, created=ts)


def openai_messages(msgs: list[OpenAIChatMessage]) -> list[dict[str, str]]:
    """Flatten OpenAI messages to harness {role, content} pairs.

    Content-part lists keep only ``{"type": "text"}`` entries; any other
    part type or a tool-bearing message fails closed — the gated pipeline
    has no tool channel and silently dropping caller content is a lie.
    """
    out: list[dict[str, str]] = []
    for i, m in enumerate(msgs):
        if (
            getattr(m, "tool_calls", None) is not None
            or getattr(m, "tool_call_id", None) is not None
        ):
            raise OpenAICompatError(f"messages[{i}]: tool calls are not supported")
        content = m.content
        if content is None:
            raise OpenAICompatError(f"messages[{i}]: content is required")
        if isinstance(content, list):
            parts: list[str] = []
            for j, part in enumerate(content):
                if part.get("type") != "text":
                    raise OpenAICompatError(
                        f"messages[{i}].content[{j}]: only 'text' parts are supported",
                    )
                parts.append(str(part.get("text", "")))
            content = "".join(parts)
        out.append({"role": m.role, "content": content})
    return out


def _resolve_openai_link(
    model: str, ext: OpenAIFx1 | None, hdrs: dict[str, str]
) -> tuple[str, list[str], str | None, ByokOverride | None]:
    """Backend resolution shared by the chat and responses translators —
    ``fx1.backend`` > ``X-Fx1-Backend`` > a ``model`` naming a backend >
    ``byok`` when BYOK headers are present > ``hosted_k3``. Returns
    ``(backend, fallbacks, checkpoint_dir, byok)``."""
    byok_headers = hdrs.get("x-fx1-byok-base-url")
    backend = (
        (ext.backend if ext is not None else None)
        or hdrs.get("x-fx1-backend")
        or (model if model in OPENAI_BACKENDS else None)
        # BYOK headers present and no explicit backend → the model string is
        # the upstream model (e.g. "gpt-4o"), the link is byok.
        or ("byok" if byok_headers else "hosted_k3")
    )
    if backend not in OPENAI_BACKENDS:
        raise OpenAICompatError(f"unknown backend {backend!r}")
    fallbacks: list[str] = list(ext.fallbacks) if ext is not None else []
    if not fallbacks and hdrs.get("x-fx1-fallbacks"):
        fallbacks = [f.strip() for f in hdrs["x-fx1-fallbacks"].split(",") if f.strip()]
    checkpoint_dir = (ext.checkpoint_dir if ext is not None else None) or hdrs.get(
        "x-fx1-checkpoint-dir"
    )
    byok = ext.byok if ext is not None else None
    if byok is None and byok_headers:
        api_key = hdrs.get("x-fx1-byok-api-key")
        byok_model = hdrs.get("x-fx1-byok-model") or (
            model if model not in OPENAI_BACKENDS else None
        )
        if not api_key:
            raise OpenAICompatError("X-Fx1-Byok-Base-Url requires X-Fx1-Byok-Api-Key")
        if not byok_model:
            raise OpenAICompatError("byok needs a model — set X-Fx1-Byok-Model or body.model")
        byok = ByokOverride(base_url=byok_headers, api_key=api_key, model=byok_model)
    return backend, fallbacks, checkpoint_dir, byok


def openai_to_kwargs(
    body: OpenAIChatRequest, headers: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Translate an OpenAI request into ``complete`` kwargs.

    Returns the keyword set both surfaces consume: the API wraps it in a
    ``CompleteRequest`` (model-level validation → 422), the SDK passes it
    to ``Fx1Harness.complete``. Keys: ``messages`` (``{role, content}``
    dicts), ``backend``, ``fallbacks``, ``byok`` (a validated dict),
    ``checkpoint_dir``, ``receipt_hashes``, ``timeout_s``, ``temperature``,
    ``top_p``, ``max_tokens``, ``seed``.

    Backend resolution order: ``fx1.backend`` > ``X-Fx1-Backend`` header >
    a ``model`` naming a backend > ``hosted_k3``. BYOK credentials come
    from ``fx1.byok`` or the ``X-Fx1-Byok-*`` headers — ``X-Fx1-Byok-Model``
    defaults to the request ``model`` when it isn't a backend name.
    """
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    ext = body.fx1
    backend, fallbacks, checkpoint_dir, byok = _resolve_openai_link(body.model, ext, hdrs)
    return {
        "backend": backend,
        "messages": openai_messages(body.messages),
        "checkpoint_dir": checkpoint_dir,
        "receipt_hashes": ext.receipt_hashes if ext is not None else None,
        "timeout_s": ext.timeout_s if ext is not None else None,
        "fallbacks": fallbacks,
        "byok": byok.model_dump() if byok is not None else None,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": (
            body.max_completion_tokens
            if body.max_completion_tokens is not None
            else body.max_tokens
        ),
        "seed": body.seed,
        "stop": (
            [body.stop]
            if isinstance(body.stop, str)
            else (list(body.stop) or None)
            if body.stop is not None
            else None
        ),
        "presence_penalty": body.presence_penalty,
        "frequency_penalty": body.frequency_penalty,
        "logit_bias": body.logit_bias,
        "user": body.user,
        "metadata": body.metadata,
        "service_tier": body.service_tier,
        "reasoning_effort": body.reasoning_effort,
        "prompt_cache_key": body.prompt_cache_key,
    }


def validate_response_format(rf: dict[str, Any] | None, content: str) -> None:
    """Post-validate a gated completion against a response_format dict.

    The harness can't constrain-decode arbitrary providers (BYOK),
    so the gate's second pass verifies the returned text: a
    non-conforming output is a provider-side failure — 502 /
    ``format_violation``, never silently shipped and never pinned
    into an idempotency record.
    """
    if not rf:
        return
    rtype = rf.get("type", "text")
    if rtype == "text":
        return
    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAICompatError(
            f"model output is not valid JSON under response_format {rtype!r}: {exc}",
            status=502,
            code="format_violation",
        ) from exc
    if rtype == "json_object" and not isinstance(parsed, dict):
        raise OpenAICompatError(
            f"model output under json_object must be a JSON object, got {type(parsed).__name__}",
            status=502,
            code="format_violation",
        )
    if rtype == "json_schema":
        schema = rf["json_schema"]["schema"]
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            raise OpenAICompatError(
                "model output failed the declared json_schema: "
                f"{exc.message} (at {list(exc.absolute_path)})",
                status=502,
                code="format_violation",
            ) from exc


def validate_openai_output(body: OpenAIChatRequest, content: str) -> None:
    """Post-validate a gated chat completion — thin shim over
    :func:`validate_response_format` on ``body.response_format``."""
    validate_response_format(body.response_format, content)


def openai_usage(usage: dict[str, int] | None) -> dict[str, int] | None:
    """Usage dict in the OpenAI key set — the BYOK/hosted backends already
    report prompt/completion/total; anything else passes through so the
    evidence field stays faithful."""
    if not isinstance(usage, dict):
        return None
    return {k: v for k, v in usage.items() if isinstance(v, int)}


def openai_envelope(
    *,
    cid: str,
    content: str | list[str],
    backend: str,
    model: str | None = None,
    usage: dict[str, int] | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """A gated result → the `chat.completion` envelope. `model` reports
    the serving link's own model id (or the backend name); the completion
    id mints the `chatcmpl-` handle. ``content`` accepts the ``n>1``
    choice list — one entry per completion, index-ordered."""
    contents = [content] if isinstance(content, str) else list(content)
    return {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
            for i, text in enumerate(contents)
        ],
        "usage": openai_usage(usage),
    }


def _text_pieces(text: str) -> Iterator[str]:
    """~64-char delta pieces on whitespace boundaries — the shared
    chunking for ``chat.completion.chunk`` deltas and ``output_text``
    response events."""
    pos = 0
    while pos < len(text):
        end = min(pos + 64, len(text))
        if end < len(text):
            sp = text.rfind(" ", pos, end)
            if sp > pos:
                end = sp + 1
        yield text[pos:end]
        pos = end


def openai_chunks(
    *,
    text: str | list[str],
    backend: str,
    model: str | None = None,
    cid: str,
    include_usage: bool = False,
    usage: dict[str, int] | None = None,
    created: int | None = None,
) -> Iterator[dict[str, Any]]:
    """`chat.completion.chunk` payloads over gated text.

    The content is already past the honesty gate (it is the gated
    response), so delta chunking is a presentation choice — ~64-char
    pieces on whitespace boundaries so streaming clients render
    incrementally. ``include_usage`` adds the OpenAI usage chunk
    (``choices=[]``, usage set) before the wire's ``[DONE]`` marker —
    which is emitted by the serializer, not this generator.

    ``text`` accepts the ``n>1`` choice list: each index emits its own
    role delta, content deltas, and ``finish_reason`` frame — grouped
    per index (spec-legal; ``choices[].index`` disambiguates).
    """
    texts = [text] if isinstance(text, str) else list(text)
    base: dict[str, Any] = {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
    }
    for i, choice_text in enumerate(texts):
        first = dict(base)
        first["choices"] = [{"index": i, "delta": {"role": "assistant"}, "finish_reason": None}]
        yield first

        for piece in _text_pieces(choice_text):
            frame = dict(base)
            frame["choices"] = [{"index": i, "delta": {"content": piece}, "finish_reason": None}]
            yield frame

        last = dict(base)
        last["choices"] = [{"index": i, "delta": {}, "finish_reason": "stop"}]
        yield last

    if include_usage:
        usage_frame = dict(base)
        usage_frame["choices"] = []
        usage_frame["usage"] = openai_usage(usage)
        yield usage_frame


# --- Responses API surface ----------------------------------------------------
# POST /v1/responses — OpenAI's canonical surface. Same gated pipeline, same
# translation layer, same fail-closed rule: a field the pipeline can't honor
# is a 400, never a silent drop. ``store`` is honored: it governs the
# retrieval index behind ``GET|DELETE /v1/responses/{id}`` (the audit ledger
# records every call regardless — retrieval is a convenience surface, not
# the evidence).
RESPONSES_UNSUPPORTED = (
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "truncation",
    "include",
    "background",
    "previous_response_id",
    # chat-completions fields that don't exist on this surface — refuse
    # rather than drop so a caller's intent never evaporates
    "n",
    "stop",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "messages",
    "stream_options",
    "response_format",
    "max_tokens",
    "seed",
    "echo",
    "suffix",
    "best_of",
    "modalities",
    "audio",
    "prediction",
    "web_search_options",
    "functions",
    "function_call",
)

# Item types inside ``input[]`` that a text-only gated pipeline cannot honor.
RESPONSE_ITEM_TYPES_REFUSED = frozenset(
    {
        "item_reference",
        "function_call",
        "function_call_output",
        "reasoning",
        "web_search_call",
        "file_search_call",
        "computer_call",
        "computer_call_output",
        "code_interpreter_call",
        "image_generation_call",
        "local_shell_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
    }
)

# Content-part types inside a message item the pipeline accepts.
RESPONSE_PART_TYPES = frozenset({"input_text", "output_text"})

RESPONSE_ROLES = frozenset({"user", "assistant", "system", "developer"})


class OpenAIResponseRequest(_Model):
    """POST /v1/responses body — the Responses surface over the same
    gated pipeline. ``input`` is one string or a list of message items;
    ``instructions`` prepends a system message. ``reasoning.effort`` maps
    to the decode hint; ``text.format`` maps to the post-validated
    ``response_format`` channel; ``user``/``safety_identifier`` stamp the
    audit record."""

    model_config = ConfigDict(extra="allow")

    model: str = "fx1"
    input: str | list[dict[str, Any]]
    instructions: str | None = Field(default=None, max_length=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, gt=0, le=262144)
    stream: bool = False
    store: bool | None = None
    metadata: dict[str, str] | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    user: str | None = Field(default=None, max_length=512)
    safety_identifier: str | None = Field(default=None, max_length=512)
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _response_valid(self) -> OpenAIResponseRequest:
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input must not be empty")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input must not be empty")
        if self.metadata is not None:
            if len(self.metadata) > 16:
                raise ValueError("metadata accepts at most 16 entries")
            for k, v in self.metadata.items():
                if len(k) > 64 or len(v) > 512:
                    raise ValueError("metadata keys are ≤64 chars, values ≤512")
        if self.reasoning is not None:
            extra = set(self.reasoning) - {"effort"}
            if extra:
                raise ValueError(
                    f"unsupported reasoning keys: {sorted(extra)} "
                    "(only 'effort' maps onto the gated decode hints)"
                )
            effort = self.reasoning.get("effort")
            if effort is not None and effort not in (
                "none",
                "minimal",
                "low",
                "medium",
                "high",
            ):
                raise ValueError(
                    f"reasoning.effort must be none|minimal|low|medium|high, got {effort!r}"
                )
        if self.text is not None:
            extra_t = set(self.text) - {"format", "verbosity"}
            if extra_t:
                raise ValueError(f"unsupported text keys: {sorted(extra_t)}")
            fmt = self.text.get("format")
            if fmt is not None:
                if not isinstance(fmt, dict):
                    raise ValueError("text.format must be an object")
                ftype = fmt.get("type", "text")
                if ftype not in OPENAI_RESPONSE_FORMATS:
                    raise ValueError(
                        "text.format.type must be one of "
                        f"{sorted(OPENAI_RESPONSE_FORMATS)}; got {ftype!r}"
                    )
                if ftype == "json_schema":
                    schema = fmt.get("schema")
                    if not isinstance(schema, dict):
                        raise ValueError(
                            "text.format json_schema needs "
                            "{type: 'json_schema', name?, schema: {...}}"
                        )
                    try:
                        jsonschema.validators.validator_for(schema).check_schema(schema)
                    except jsonschema.SchemaError as exc:
                        raise ValueError(
                            f"text.format json_schema is not a valid schema: {exc.message}"
                        ) from exc
        present = [f for f in RESPONSES_UNSUPPORTED if getattr(self, f, None) is not None]
        extra_bad = sorted(f for f in RESPONSES_UNSUPPORTED if f in (self.__pydantic_extra__ or {}))
        bad = sorted(set(present) | set(extra_bad))
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


def response_input_to_messages(
    input_: str | list[dict[str, Any]], instructions: str | None
) -> list[dict[str, str]]:
    """Flatten a Responses ``input`` + ``instructions`` into harness
    ``{role, content}`` pairs. ``developer`` items map to ``system``;
    every non-message item type and non-text content part fails
    closed."""
    msgs: list[dict[str, str]] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})
    if isinstance(input_, str):
        msgs.append({"role": "user", "content": input_})
        return msgs
    for i, item in enumerate(input_):
        if not isinstance(item, dict):
            raise OpenAICompatError(f"input[{i}]: items must be objects")
        itype = item.get("type")
        if itype in RESPONSE_ITEM_TYPES_REFUSED:
            raise OpenAICompatError(f"input[{i}]: {itype!r} items are not supported")
        role = item.get("role")
        if itype not in (None, "message") or role is None:
            raise OpenAICompatError(f"input[{i}]: only message items with a role are supported")
        if role not in RESPONSE_ROLES:
            raise OpenAICompatError(f"input[{i}]: unknown role {role!r}")
        content = item.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    raise OpenAICompatError(f"input[{i}].content[{j}]: parts must be objects")
                ptype = part.get("type")
                if ptype not in RESPONSE_PART_TYPES:
                    raise OpenAICompatError(
                        f"input[{i}].content[{j}]: only text parts are supported, got {ptype!r}"
                    )
                if ptype == "output_text" and role != "assistant":
                    raise OpenAICompatError(
                        f"input[{i}].content[{j}]: output_text belongs to assistant items"
                    )
                parts.append(str(part.get("text", "")))
            text = "".join(parts)
        else:
            raise OpenAICompatError(f"input[{i}]: content is required")
        msgs.append({"role": "system" if role == "developer" else role, "content": text})
    if not any(m["role"] != "system" for m in msgs):
        raise OpenAICompatError("input carried no user/assistant turn")
    return msgs


def response_text_format(body: OpenAIResponseRequest) -> dict[str, Any] | None:
    """``text.format`` → the chat-shape ``response_format`` dict
    :func:`validate_response_format` consumes (or ``None``)."""
    if not body.text:
        return None
    fmt = body.text.get("format")
    if not isinstance(fmt, dict):
        return None
    ftype = fmt.get("type", "text")
    if ftype == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": fmt.get("name"), "schema": fmt["schema"]},
        }
    return {"type": ftype}


def response_to_kwargs(
    body: OpenAIResponseRequest, headers: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Translate a Responses request into ``complete`` kwargs — the same
    backend-resolution order and the same extension/headers as the chat
    surface. ``max_output_tokens`` lands on ``max_tokens``;
    ``reasoning.effort`` on ``reasoning_effort``; the audit stamp takes
    ``user`` or ``safety_identifier``."""
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    ext = body.fx1
    backend, fallbacks, checkpoint_dir, byok = _resolve_openai_link(body.model, ext, hdrs)
    effort = (body.reasoning or {}).get("effort")
    return {
        "backend": backend,
        "messages": response_input_to_messages(body.input, body.instructions),
        "checkpoint_dir": checkpoint_dir,
        "receipt_hashes": ext.receipt_hashes if ext is not None else None,
        "timeout_s": ext.timeout_s if ext is not None else None,
        "fallbacks": fallbacks,
        "byok": byok.model_dump() if byok is not None else None,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_output_tokens,
        "user": body.user or body.safety_identifier,
        "metadata": body.metadata,
        "service_tier": body.service_tier,
        "reasoning_effort": effort,
    }


def _response_echoes(body: OpenAIResponseRequest) -> dict[str, Any]:
    """The request fields a faithful ``response`` object echoes back."""
    return {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_output_tokens": body.max_output_tokens,
        "metadata": body.metadata,
        "instructions": body.instructions,
        "service_tier": body.service_tier,
        "reasoning": body.reasoning,
        "text": body.text,
        # fields we refuse on input are echoed as their honest constants;
        # `store` echoes the actual knob — the retrieval index honors it
        "store": body.store is not False,
        "tools": [],
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "truncation": "disabled",
    }


def openai_response_object(
    *,
    rid: str,
    item_id: str,
    content: str,
    body: OpenAIResponseRequest,
    model: str | None,
    usage: dict[str, int] | None,
    status: str = "completed",
    created: int | None = None,
) -> dict[str, Any]:
    """A gated result → the ``response`` object. ``output`` carries one
    ``message`` item with one ``output_text`` part; ``usage`` maps the
    provider's counts onto input/output/total (``None`` when the backend
    reports nothing — never fabricated). ``status`` is ``in_progress``
    only inside the pre-completion stream events."""
    resp_usage: dict[str, int] | None = None
    if isinstance(usage, dict):
        it = usage.get("prompt_tokens")
        ot = usage.get("completion_tokens")
        tt = usage.get("total_tokens")
        if isinstance(it, int) or isinstance(ot, int) or isinstance(tt, int):
            i_v = it if isinstance(it, int) else 0
            o_v = ot if isinstance(ot, int) else 0
            resp_usage = {
                "input_tokens": i_v,
                "output_tokens": o_v,
                "total_tokens": tt if isinstance(tt, int) else i_v + o_v,
            }
    output = (
        [
            {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ]
        if status == "completed"
        else []
    )
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()) if created is None else created,
        "status": status,
        "model": model or body.model,
        "output": output,
        "usage": resp_usage,
        "error": None,
        "incomplete_details": None,
        **_response_echoes(body),
    }


def openai_response_events(
    *,
    text: str,
    rid: str,
    item_id: str,
    body: OpenAIResponseRequest,
    model: str | None,
    usage: dict[str, int] | None,
    created: int | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """The Responses SSE event sequence over gated text — the core grammar
    a streaming client needs: ``response.created``/``in_progress``, the
    output-item lifecycle, ``output_text.delta`` frames (the shared
    ~64-char splitter), and ``response.completed`` carrying the full
    response object with usage. Each ``(event, payload)`` pair serializes
    as an SSE ``event:`` + ``data:`` frame."""
    created_obj = openai_response_object(
        rid=rid,
        item_id=item_id,
        content="",
        body=body,
        model=model,
        usage=None,
        status="in_progress",
        created=created,
    )
    yield "response.created", {"type": "response.created", "response": created_obj}
    yield (
        "response.in_progress",
        {
            "type": "response.in_progress",
            "response": created_obj,
        },
    )
    yield (
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
    )
    yield (
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )
    for piece in _text_pieces(text):
        yield (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
            },
        )
    yield (
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
        },
    )
    yield (
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": text, "annotations": []},
        },
    )
    yield (
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        },
    )
    yield (
        "response.completed",
        {
            "type": "response.completed",
            "response": openai_response_object(
                rid=rid,
                item_id=item_id,
                content=text,
                body=body,
                model=model,
                usage=usage,
                status="completed",
                created=created,
            ),
        },
    )


# ---- /v1/files + /v1/batches ------------------------------------------------
# The async-batch surface: ``POST /v1/files`` takes the request JSONL
# (multipart, ``purpose="batch"``), ``POST /v1/batches`` runs it through the
# gated pipeline as one tracked unit, and ``GET /v1/files/{id}/content``
# returns the output JSONL. Same fail-closed rule as the rest of /v1 —
# shapes the pipeline can't honor refuse at submit; per-line request errors
# land in the output file with their OpenAI error body, never silently.

OPENAI_BATCH_ENDPOINTS = frozenset({"/v1/chat/completions", "/v1/responses"})
"""Endpoints a batch may target — one per batch, declared up front."""

OPENAI_BATCH_LINE_MAX = 1024
"""Max request lines per batch file."""

OPENAI_FILE_BYTES_MAX = 8 << 20
"""Max upload size (8 MiB)."""

OPENAI_FILE_PURPOSE_ACCEPT = "batch"
"""The only upload purpose the harness serves — batch input JSONL."""


class OpenAIBatchRequest(_Model):
    """``POST /v1/batches`` body."""

    model_config = ConfigDict(extra="allow")
    input_file_id: str = Field(min_length=1, max_length=64)
    endpoint: Literal["/v1/chat/completions", "/v1/responses"]
    # only "24h" exists on the real surface; anything else refuses (422)
    completion_window: Literal["24h"] = "24h"
    metadata: dict[str, str] | None = None

    @field_validator("metadata")
    @classmethod
    def _meta_bounds(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is not None and len(v) > 16:
            raise ValueError("metadata must have <= 16 keys")
        return v


def batch_line_shape(line: Any, *, endpoint: str, lineno: int) -> dict[str, Any]:
    """Validate one input-file line into ``{custom_id, body}``.

    Line-shape errors are submit-time failures (400) — a batch whose input
    can't be trusted never starts, rather than half-running a corrupt file.
    Per-line *request* errors (a valid line whose body fails the endpoint's
    own validation) run through the pipeline and land in the output file.
    """
    where = f"line {lineno}"
    if not isinstance(line, dict):
        raise OpenAICompatError(f"{where}: must be a JSON object")
    custom_id = line.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id.strip() or len(custom_id) > 64:
        raise OpenAICompatError(f"{where}: custom_id must be a non-empty string <= 64 chars")
    if line.get("method") != "POST":
        raise OpenAICompatError(f"{where}: method must be 'POST', got {line.get('method')!r}")
    url = line.get("url")
    if url != endpoint:
        raise OpenAICompatError(
            f"{where}: url {url!r} does not match the batch endpoint {endpoint!r}"
        )
    body = line.get("body")
    if not isinstance(body, dict):
        raise OpenAICompatError(f"{where}: body must be a JSON object")
    return {"custom_id": custom_id, "body": body}


def batch_line_body(
    line: dict[str, Any], endpoint: str
) -> OpenAIChatRequest | OpenAIResponseRequest:
    """Parse a line's ``body`` into the endpoint's request model — the same
    validation object the wire route uses, so a batch line can never carry
    a request the live route would refuse differently."""
    try:
        if endpoint == "/v1/responses":
            return OpenAIResponseRequest.model_validate(line["body"])
        return OpenAIChatRequest.model_validate(line["body"])
    except ValueError as exc:
        raise OpenAICompatError(f"invalid request body: {exc}") from exc


def file_object(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The OpenAI ``file`` envelope for a stored record."""
    return {
        "id": rec["file_id"],
        "object": "file",
        "purpose": rec["purpose"],
        "filename": rec["filename"],
        "bytes": rec["size"],
        "created_at": rec["created_at"],
        "status": "processed",
    }


def batch_object(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The OpenAI ``batch`` envelope for a stored record."""
    return {
        "id": rec["batch_id"],
        "object": "batch",
        "endpoint": rec["endpoint"],
        "errors": rec.get("errors"),
        "input_file_id": rec["input_file_id"],
        "completion_window": rec["completion_window"],
        "status": rec["status"],
        "output_file_id": rec.get("output_file_id"),
        "error_file_id": rec.get("error_file_id"),
        "created_at": rec["created_at"],
        "in_progress_at": rec.get("in_progress_at"),
        "expires_at": rec.get("expires_at"),
        "finalizing_at": rec.get("finalizing_at"),
        "completed_at": rec.get("completed_at"),
        "failed_at": rec.get("failed_at"),
        "expired_at": rec.get("expired_at"),
        "cancelling_at": rec.get("cancelling_at"),
        "cancelled_at": rec.get("cancelled_at"),
        "request_counts": dict(rec["request_counts"]),
        "metadata": rec.get("metadata"),
    }


def batch_output_line(
    *,
    custom_id: str,
    status_code: int,
    body: dict[str, Any],
    rid: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One output-file line — the OpenAI batch result shape."""
    return {
        "id": f"batch_req_{rid}",
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "request_id": f"req_{rid}",
            "body": body,
        },
        "error": error,
    }


# ---- retrieval index (GET /v1/chat/completions/{id}, GET|DELETE /v1/responses/{id}) ----


class OpenAIEnvelopeStore:
    """Bounded ``id → envelope`` index backing the ``/v1`` retrieval
    routes. ``store=false`` keeps a call out of this index — the
    completion log still records it; retrieval is a convenience surface,
    not the evidence. Oldest entries evict at ``cap`` (insertion order —
    a re-put refreshes the id's position)."""

    def __init__(self, cap: int = 256) -> None:
        if cap < 1:
            raise ValueError(f"store cap must be >= 1, got {cap}")
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}

    def put(self, envelope: dict[str, Any]) -> None:
        eid = envelope.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("envelope carries no string 'id'")
        with self._lock:
            self._items.pop(eid, None)
            self._items[eid] = envelope
            while len(self._items) > self._cap:
                self._items.pop(next(iter(self._items)))

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        with self._lock:
            env = self._items.get(envelope_id)
            return dict(env) if env is not None else None

    def delete(self, envelope_id: str) -> bool:
        with self._lock:
            return self._items.pop(envelope_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
