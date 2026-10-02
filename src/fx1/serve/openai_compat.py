"""OpenAI-compatible ingress — the shared translation layer.

The ``/v1`` HTTP routes (``fx1.serve.api``) and the in-process SDK surface
(``fx1.sdk.Fx1Harness.openai_chat``) translate through THIS module — one
pydantic validation object, one backend-resolution order, one error
taxonomy — so the wire and the weights-direct path cannot drift apart.

- *Fail-closed surface* — every OpenAI feature the gated pipeline cannot
  honor (tools, ``n != 1``, ``logprobs``, non-text ``response_format``,
  content parts other than ``text``…) is rejected by
  :class:`OpenAICompatError`, never silently dropped.
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
    "ByokOverride",
    "OpenAIChatMessage",
    "OpenAIChatResponse",
    "OpenAIChatChoice",
    "OpenAICompatError",
    "OpenAIFx1",
    "OpenAIModel",
    "OpenAIModelList",
    "is_openai_path",
    "openai_chunks",
    "openai_envelope",
    "openai_error_body",
    "openai_messages",
    "openai_model",
    "openai_models",
    "openai_to_kwargs",
    "openai_usage",
    "validate_openai_output",
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
OPENAI_UNSUPPORTED = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "modalities",
    "audio",
    "prediction",
    "reasoning_effort",
    "service_tier",
    "store",
    "metadata",
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
    seed: int | None = Field(default=None, ge=0)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int = 1
    response_format: dict[str, Any] | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _openai_valid(self) -> OpenAIChatRequest:
        if self.n != 1:
            raise ValueError("n must be 1 — the gated pipeline serves one completion")
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
    byok_headers = hdrs.get("x-fx1-byok-base-url")
    backend = (
        (ext.backend if ext is not None else None)
        or hdrs.get("x-fx1-backend")
        or (body.model if body.model in OPENAI_BACKENDS else None)
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
        model = hdrs.get("x-fx1-byok-model") or (
            body.model if body.model not in OPENAI_BACKENDS else None
        )
        if not api_key:
            raise OpenAICompatError("X-Fx1-Byok-Base-Url requires X-Fx1-Byok-Api-Key")
        if not model:
            raise OpenAICompatError("byok needs a model — set X-Fx1-Byok-Model or body.model")
        byok = ByokOverride(base_url=byok_headers, api_key=api_key, model=model)
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
        "max_tokens": body.max_tokens,
        "seed": body.seed,
    }


def validate_openai_output(body: OpenAIChatRequest, content: str) -> None:
    """Post-validate a gated completion against ``response_format``.

    The harness can't constrain-decode arbitrary providers (BYOK),
    so the gate's second pass verifies the returned text: a
    non-conforming output is a provider-side failure — 502 /
    ``format_violation``, never silently shipped and never pinned
    into an idempotency record.
    """
    rf = body.response_format
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
    content: str,
    backend: str,
    model: str | None = None,
    usage: dict[str, int] | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """A gated result → the `chat.completion` envelope. `model` reports
    the serving link's own model id (or the backend name); the completion
    id mints the `chatcmpl-` handle."""
    return {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": openai_usage(usage),
    }


def openai_chunks(
    *,
    text: str,
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
    """
    base: dict[str, Any] = {
        "id": f"chatcmpl-{cid}",
        "object": "chat.completion.chunk",
        "created": int(time.time()) if created is None else created,
        "model": model or backend,
        "system_fingerprint": backend,
    }
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    yield first

    pos = 0
    while pos < len(text):
        end = min(pos + 64, len(text))
        if end < len(text):
            sp = text.rfind(" ", pos, end)
            if sp > pos:
                end = sp + 1
        piece = text[pos:end]
        pos = end
        frame = dict(base)
        frame["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
        yield frame

    last = dict(base)
    last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    yield last

    if include_usage:
        usage_frame = dict(base)
        usage_frame["choices"] = []
        usage_frame["usage"] = openai_usage(usage)
        yield usage_frame
