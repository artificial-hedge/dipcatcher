"""Anthropic Messages API compatibility — ``POST /v1/messages``.

One direction: Anthropic-shaped requests translate into the gated
pipeline's OpenAI core (``anthropic_to_openai`` → ``OpenAIChatRequest``
→ the shared ``_openai_chat_core`` / ``Fx1Harness.openai_chat`` path),
then the ``chat.completion`` envelope translates out to Anthropic's
``message`` object (``anthropic_envelope``) or the
``message_start``/``content_block_*``/``message_delta``/``message_stop``
SSE grammar (``anthropic_events``).

Fail-closed like the OpenAI surface: a request field the gated pipeline
cannot honor is a ``400 invalid_request_error``, never silently dropped
(``top_k``, ``thinking``, ``service_tier``, ``mcp_servers``,
``container``, ``context_management``, ``cache_control``, image/document
blocks, …). Anthropic's transcript contract — ``user`` first, strict
``user``/``assistant`` alternation — validates up front so the model
never sees an out-of-contract history.

There is no Anthropic retrieval twin (no ``GET /v1/messages/{id}``), so
the translated request pins ``store=False``: the completion log still
records the call (``GET /harness/completions/{id}`` / ``…/receipt``),
but nothing pollutes the ``/v1/chat/completions`` retrieval index.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from fx1.serve.openai_compat import (
    OpenAICompatError,
    OpenAIFx1,
    _Model,
    _text_pieces,
)
from fx1.serve.webhooks import check_callback_url

__all__ = [
    "ANTHROPIC_UNSUPPORTED",
    "AnthropicBatchCounts",
    "AnthropicBatchCreate",
    "AnthropicBatchItem",
    "AnthropicCountTokensRequest",
    "AnthropicMessage",
    "AnthropicMessageObject",
    "AnthropicMessagesRequest",
    "AnthropicMetadata",
    "AnthropicTool",
    "AnthropicToolChoice",
    "anthropic_batch_object",
    "anthropic_batch_result",
    "anthropic_count_messages",
    "anthropic_envelope",
    "anthropic_error_body",
    "anthropic_events",
    "anthropic_model_object",
    "anthropic_to_openai",
]

# Status → Anthropic ``error.type`` (the SDK reads this field to pick its
# exception class). Anything unmapped falls to api_error/invalid_request.
ANTHROPIC_ERR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    409: "invalid_request_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}

# Request fields Anthropic documents but the gated pipeline cannot honor.
# Refusing beats silently ignoring — the caller learns the truth at the
# boundary, not after a hallucinated assumption.
ANTHROPIC_UNSUPPORTED = frozenset(
    {
        "top_k",
        "thinking",
        "service_tier",
        "mcp_servers",
        "container",
        "container_upload",
        "context_management",
        "context_editing",
        "inference_geo",
        "speed",
        "safety_identifier",
        "agent",
        "effort",
    }
)

# Content block types with no channel in the gated pipeline.
_BLOCK_UNSUPPORTED = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "mcp_tool_use",
        "mcp_tool_result",
        "thinking",
        "redacted_thinking",
        "citations",
    }
)

_FINISH_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def anthropic_error_body(message: str, status: int) -> dict[str, Any]:
    """The Anthropic error envelope — ``{type: "error", error: {type, message}}``."""
    return {
        "type": "error",
        "error": {
            "type": ANTHROPIC_ERR_TYPES.get(
                status, "api_error" if status >= 500 else "invalid_request_error"
            ),
            "message": message,
        },
    }


class AnthropicUsage(_Model):
    """Anthropic ``usage`` — input/output token counts plus the cache
    fields (always 0 here — the gated pipeline has no prompt cache;
    reporting zeros is the honest contract)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessageObject(_Model):
    """The ``POST /v1/messages`` response — Anthropic's ``message``
    object. ``content`` blocks are loose dicts (``text``/``tool_use``);
    ``stop_reason`` is the OpenAI finish verdict translated
    (``end_turn``/``max_tokens``/``tool_use``/``refusal``)."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[dict[str, Any]]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"] | None
    stop_sequence: str | None = None
    usage: AnthropicUsage


class AnthropicMetadata(_Model):
    """Anthropic's ``metadata`` object — ``user_id`` is the only field the
    contract defines (maps to OpenAI's ``user`` audit stamp)."""

    user_id: str | None = Field(default=None, max_length=512)


class AnthropicTool(_Model):
    """One Anthropic ``tools[]`` entry — ``input_schema`` is the JSON
    Schema the model fills into a ``tool_use`` block's ``input``."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str | None = Field(default=None, max_length=4096)
    input_schema: dict[str, Any]
    type: Literal["custom"] = "custom"

    @model_validator(mode="after")
    def _schema_object(self) -> AnthropicTool:
        if self.input_schema.get("type") != "object":
            raise ValueError('tools[].input_schema must declare {"type": "object"}')
        return self


class AnthropicToolChoice(_Model):
    """Anthropic ``tool_choice``: ``auto``/``any``/``tool``/``none`` —
    ``tool`` pins one tool by ``name``."""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None

    @model_validator(mode="after")
    def _choice_valid(self) -> AnthropicToolChoice:
        if self.type == "tool" and not self.name:
            raise ValueError("tool_choice type 'tool' requires a name")
        if self.type != "tool" and self.name is not None:
            raise ValueError("tool_choice.name only applies to type 'tool'")
        return self


class AnthropicMessage(_Model):
    """One Anthropic transcript turn — ``role`` is ``user`` or
    ``assistant``; ``content`` is a string or a block list. Supported
    block types: ``text`` (any turn), ``tool_use`` (assistant turns —
    a prior round's calls), ``tool_result`` (user turns — the tool's
    answer). Every other declared block type is refused, not dropped."""

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]

    @model_validator(mode="after")
    def _blocks_valid(  # NOSONAR(S3776) — per-block shape dispatch is inherently branchy
        self,
    ) -> AnthropicMessage:
        if isinstance(self.content, str):
            return self
        if not self.content:
            raise ValueError("messages[].content must be a non-empty string or block list")
        seen_tool_use = False
        for block in self.content:
            btype = block.get("type")
            if btype == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError("text blocks need a string 'text'")
            elif btype == "tool_use":
                if self.role != "assistant":
                    raise ValueError("tool_use blocks belong on assistant turns")
                if not isinstance(block.get("id"), str) or not block.get("id"):
                    raise ValueError("tool_use blocks need an 'id'")
                if not isinstance(block.get("name"), str) or not block.get("name"):
                    raise ValueError("tool_use blocks need a 'name'")
                if "input" not in block:
                    raise ValueError("tool_use blocks need an 'input'")
                seen_tool_use = True
            elif btype == "tool_result":
                if self.role != "user":
                    raise ValueError("tool_result blocks belong on user turns")
                if not isinstance(block.get("tool_use_id"), str) or not block.get("tool_use_id"):
                    raise ValueError("tool_result blocks need a 'tool_use_id'")
            elif btype in _BLOCK_UNSUPPORTED:
                raise ValueError(
                    f"content block type {btype!r} is not supported by the "
                    "gated pipeline (text / tool_use / tool_result only)"
                )
            elif btype is None:
                raise ValueError("content blocks need a 'type'")
            else:
                raise ValueError(f"unknown content block type {btype!r}")
            if "cache_control" in block:
                raise ValueError("cache_control is not supported by the gated pipeline")
        if seen_tool_use and self.role == "user":
            raise ValueError("tool_use blocks belong on assistant turns")
        return self


class AnthropicMessagesRequest(_Model):
    """``POST /v1/messages`` body — Anthropic's create-message contract.
    ``max_tokens`` is required (the Anthropic contract, unlike OpenAI's).
    Extra fields are tolerated for SDK bookkeeping keys but the
    documented-but-unsupported knobs in ``ANTHROPIC_UNSUPPORTED`` refuse
    (a 400, never a silent ignore)."""

    model_config = ConfigDict(extra="allow")

    model: str = "fx1"
    messages: list[AnthropicMessage] = Field(min_length=1, max_length=512)
    max_tokens: int = Field(gt=0, le=262144)
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop_sequences: list[str] | None = Field(default=None, max_length=4)
    stream: bool = False
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    metadata: AnthropicMetadata | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _anthropic_valid(  # NOSONAR(S3776) — contract validator walks every field
        self,
    ) -> AnthropicMessagesRequest:
        if self.stop_sequences is not None and any(
            not isinstance(s, str) or not 1 <= len(s) <= 512 for s in self.stop_sequences
        ):
            raise ValueError("stop_sequences must be 1–512 char strings")
        _anthropic_message_contract(self.messages, self.system, self.tools, self.tool_choice)
        bad = sorted(
            f
            for f in ANTHROPIC_UNSUPPORTED
            if f in (self.__pydantic_extra__ or {}) or getattr(self, f, None) is not None
        )
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


def _anthropic_message_contract(
    messages: list[AnthropicMessage],
    system: str | list[dict[str, Any]] | None,
    tools: list[AnthropicTool] | None,
    tool_choice: AnthropicToolChoice | None,
) -> None:
    """The message-channel contract shared by ``/v1/messages`` and
    ``/v1/messages/count_tokens`` — user-first strict alternation,
    system shape, and tool/tool_choice coherence."""
    # Anthropic's transcript contract: user first, strict alternation.
    if messages[0].role != "user":
        raise ValueError("messages must start with a 'user' turn")
    for prev, cur in zip(messages, messages[1:], strict=False):
        if prev.role == cur.role:
            raise ValueError(
                f"messages must alternate user/assistant (two consecutive {prev.role!r} turns)"
            )
    if system is not None and not isinstance(system, str):
        if not system:
            raise ValueError("system must be a string or a non-empty text-block list")
        for block in system:
            if block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise ValueError("system blocks must be {type: 'text', text: str}")
            if "cache_control" in block:
                raise ValueError("cache_control is not supported by the gated pipeline")
    if tools is not None:
        if len(tools) > 128:
            raise ValueError("tools accepts at most 128 entries")
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError("tools[] names must be unique")
    if tool_choice is not None and not tools:
        raise ValueError("tool_choice requires a non-empty tools list")
    if (
        tool_choice is not None
        and tool_choice.type == "tool"
        and tool_choice.name not in {t.name for t in tools or []}
    ):
        raise ValueError(f"tool_choice names {tool_choice.name!r}, which is not in tools[]")


class AnthropicCountTokensRequest(_Model):
    """``POST /v1/messages/count_tokens`` body — Anthropic's estimate
    contract: the create-message shape minus ``max_tokens`` (the input
    channel alone is measured). Extra fields tolerate SDK bookkeeping
    keys; the documented-but-unsupported knobs in ``ANTHROPIC_UNSUPPORTED``
    refuse identically to ``/v1/messages``.

    ``tools``/``tool_choice`` validate here (the wire shape is legal)
    but the route refuses them — a provider's ``/tokenize`` sees only
    the message channel, so counting a toolful request would undercount.
    Refusing beats lying."""

    model_config = ConfigDict(extra="allow")

    model: str = "fx1"
    messages: list[AnthropicMessage] = Field(min_length=1, max_length=512)
    system: str | list[dict[str, Any]] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    fx1: OpenAIFx1 | None = None

    @model_validator(mode="after")
    def _count_valid(self) -> AnthropicCountTokensRequest:
        _anthropic_message_contract(self.messages, self.system, self.tools, self.tool_choice)
        bad = sorted(
            f
            for f in ANTHROPIC_UNSUPPORTED
            if f in (self.__pydantic_extra__ or {}) or getattr(self, f, None) is not None
        )
        if bad:
            raise ValueError(f"unsupported for the gated pipeline: {', '.join(bad)}")
        return self


def anthropic_count_messages(body: AnthropicCountTokensRequest) -> list[dict[str, Any]]:
    """The count request's message channel in OpenAI shape — system folds
    in exactly as ``anthropic_to_openai`` does, so the count matches the
    prompt a completion would consume."""
    messages = _messages_to_openai(body.messages)
    if body.system is not None:
        sys_text = _system_text(body.system)
        if sys_text:
            messages.insert(0, {"role": "system", "content": sys_text})
    return messages


def anthropic_model_object(model_id: str, *, created: int | None = None) -> dict[str, Any]:
    """Anthropic's model card: ``{type: "model", id, display_name,
    created_at}``. ``display_name`` mirrors the id honestly — the harness
    names no display names of its own; ``created_at`` is the serve
    boot stamp (``_rfc3339``), matching the ``/v1/models`` `created`."""
    return {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
        "created_at": _rfc3339(created),
    }


def _system_text(system: str | list[dict[str, Any]]) -> str:
    """Flatten the ``system`` field to one text — blocks are validated
    text-only by the request model."""
    if isinstance(system, str):
        return system
    return "\n".join(str(b["text"]) for b in system)


def _tool_result_content(block: dict[str, Any]) -> str:
    """``tool_result`` content → the tool message's text — a string is
    verbatim; a block list joins its text parts."""
    content = block.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for inner in content:
            if not isinstance(inner, dict) or inner.get("type") != "text":
                raise OpenAICompatError(
                    "tool_result content blocks must be text",
                    status=400,
                    code="invalid_request",
                )
            parts.append(str(inner.get("text", "")))
        return "\n".join(parts)
    raise OpenAICompatError(
        "tool_result content must be a string or a text-block list",
        status=400,
        code="invalid_request",
    )


def _messages_to_openai(  # NOSONAR(S3776) — one branch per Anthropic block type
    messages: list[AnthropicMessage],
) -> list[dict[str, Any]]:
    """Anthropic turns → OpenAI chat messages.

    A ``user`` turn's ``tool_result`` blocks each become a ``role: tool``
    message (OpenAI's one-message-per-answer shape); assistant ``tool_use``
    blocks fold into that turn's ``tool_calls``. Text parts join the
    turn's ``content`` — ``None`` when an assistant turn is calls-only
    (the OpenAI contract for a pure tool turn)."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg.content, str):
            out.append({"role": msg.role, "content": msg.content})
            continue
        if msg.role == "assistant":
            texts: list[str] = []
            calls: list[dict[str, Any]] = []
            for block in msg.content:
                if block["type"] == "text":
                    texts.append(str(block["text"]))
                else:  # tool_use — validated
                    try:
                        args = json.dumps(block["input"])
                    except (TypeError, ValueError) as exc:
                        raise OpenAICompatError(
                            f"tool_use input is not JSON-serializable: {exc}",
                            status=400,
                            code="invalid_request",
                        ) from exc
                    calls.append(
                        {
                            "id": str(block["id"]),
                            "type": "function",
                            "function": {"name": str(block["name"]), "arguments": args},
                        }
                    )
            turn: dict[str, Any] = {"role": "assistant"}
            if texts:
                turn["content"] = "\n".join(texts)
            if calls:
                turn["tool_calls"] = calls
            if "content" not in turn and not calls:
                raise OpenAICompatError(
                    "assistant turn has no text or tool_use blocks",
                    status=400,
                    code="invalid_request",
                )
            out.append(turn)
            continue
        # user turn — text blocks + tool_result blocks
        texts = [str(b["text"]) for b in msg.content if b["type"] == "text"]
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
        for block in msg.content:
            if block["type"] == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block["tool_use_id"]),
                        "content": _tool_result_content(block),
                    }
                )
        if not texts and not any(b["type"] == "tool_result" for b in msg.content):
            raise OpenAICompatError(
                "user turn has no text or tool_result blocks",
                status=400,
                code="invalid_request",
            )
    return out


def anthropic_to_openai(  # NOSONAR(S3776) — field-by-field wire translation
    body: AnthropicMessagesRequest,
) -> dict[str, Any]:
    """Translate ``AnthropicMessagesRequest`` → ``OpenAIChatRequest``
    kwargs — the shared gated path then validates, backends, gates, and
    meters exactly as ``/v1/chat/completions``.

    ``store=False`` is forced: Anthropic has no retrieval twin, so the
    call stays out of the ``/v1/chat/completions`` index (the completion
    log still records it — ``GET /harness/completions/{id}``)."""
    messages = _messages_to_openai(body.messages)
    if body.system is not None:
        sys_text = _system_text(body.system)
        if sys_text:
            messages.insert(0, {"role": "system", "content": sys_text})
    tools: list[dict[str, Any]] | None = None
    if body.tools:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    **({"description": t.description} if t.description is not None else {}),
                    "parameters": t.input_schema,
                },
            }
            for t in body.tools
        ]
    tool_choice: str | dict[str, Any] | None = None
    parallel: bool | None = None
    if body.tool_choice is not None:
        tc = body.tool_choice
        if tc.type == "auto":
            tool_choice = "auto"
        elif tc.type == "any":
            tool_choice = "required"
        elif tc.type == "none":
            tool_choice = "none"
        else:  # tool
            tool_choice = {"type": "function", "function": {"name": tc.name}}
        if tc.disable_parallel_tool_use:
            parallel = False
    req: dict[str, Any] = {
        "model": body.model,
        "messages": messages,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "stop": list(body.stop_sequences) if body.stop_sequences else None,
        "stream": body.stream,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel,
        "user": body.metadata.user_id if body.metadata is not None else None,
        "fx1": body.fx1,
        "store": False,
    }
    return {k: v for k, v in req.items() if v is not None}


def anthropic_envelope(  # NOSONAR(S3776) — envelope builder fans out per content block
    env: dict[str, Any], *, model: str | None = None
) -> dict[str, Any]:
    """``chat.completion`` envelope → Anthropic ``message`` object.

    The id is derived, not minted fresh — ``msg_<hex>`` carries the
    completion's ``chatcmpl-<hex>`` so the wire answer and the completion
    log record share one fingerprint."""
    choices = env.get("choices") or []
    if not choices:
        raise OpenAICompatError(
            "the gated pipeline returned no choices", status=502, code="backend_failure"
        )
    choice = choices[0]
    message = choice.get("message") or {}
    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call, dict) else None
        raw_args = fn.get("arguments") if isinstance(fn, dict) else None
        try:
            inp = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            if not isinstance(inp, dict):
                inp = {"_value": inp}
        except json.JSONDecodeError:
            inp = {"_raw": raw_args}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")) if isinstance(fn, dict) else "",
                "input": inp,
            }
        )
    finish = str(choice.get("finish_reason") or "stop")
    stop_reason = _FINISH_TO_STOP_REASON.get(finish, "end_turn")
    raw_usage = env.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    cid = str(env.get("id") or "")
    return {
        "id": f"msg_{cid.removeprefix('chatcmpl-')}"
        if cid
        else f"msg_{env.get('created', int(time.time()))}",
        "type": "message",
        "role": "assistant",
        "model": env.get("model") or model or "fx1",
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def anthropic_events(env: dict[str, Any], *, model: str | None = None) -> Iterator[dict[str, Any]]:
    """Anthropic SSE event payloads (``{"event", "data"}`` pairs) over a
    completed ``chat.completion`` envelope — the same presentation-layer
    chunking the OpenAI stream uses (``_text_pieces``), emitted as
    ``message_start`` → ``content_block_start`` → ``content_block_delta``×n
    → ``content_block_stop`` → ``message_delta`` → ``message_stop`` per
    block. Text blocks chunk on whitespace boundaries; ``tool_use``
    blocks ship their ``input`` as one ``input_json_delta`` partial_json
    (the provider's arg deltas don't survive the gated call — a single
    complete delta is the honest frame)."""
    message = anthropic_envelope(env, model=model)
    yield {
        "event": "message_start",
        "data": {
            "type": "message_start",
            "message": {
                **{k: v for k, v in message.items() if k != "usage"},
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": dict(message["usage"]),
            },
        },
    }
    yield {"event": "ping", "data": {"type": "ping"}}
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            }
            for piece in _text_pieces(block["text"]):
                yield {
                    "event": "content_block_delta",
                    "data": {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": piece},
                    },
                }
        else:  # tool_use
            yield {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            }
            yield {
                "event": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"], separators=(",", ":")),
                    },
                },
            }
        yield {
            "event": "content_block_stop",
            "data": {"type": "content_block_stop", "index": index},
        }
    yield {
        "event": "message_delta",
        "data": {
            "type": "message_delta",
            "delta": {
                "stop_reason": message["stop_reason"],
                "stop_sequence": message["stop_sequence"],
            },
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
    }
    yield {"event": "message_stop", "data": {"type": "message_stop"}}


def anthropic_sse(env: dict[str, Any], *, model: str | None = None, skip: int = 0) -> Iterator[str]:
    """Serialize ``anthropic_events`` into SSE frames — ``event:`` +
    ``data:`` per frame, ``id: <index>`` for Last-Event-ID resume parity
    with the OpenAI surface."""
    for seq, event in enumerate(anthropic_events(env, model=model)):
        if seq >= skip:
            yield (
                f"id: {seq}\nevent: {event['event']}\n"
                f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
            )


# ---------------------------------------------------------------------------
# Message Batches — POST /v1/messages/batches
# ---------------------------------------------------------------------------


class AnthropicBatchItem(_Model):
    """One ``requests[]`` element of ``POST /v1/messages/batches`` —
    ``custom_id`` (unique within the batch, the result-line join key) and
    the full ``params`` a create call takes. ``stream`` refuses at
    validation: the batch surface has no streaming leg."""

    model_config = ConfigDict(extra="allow")

    custom_id: str = Field(min_length=1, max_length=256)
    params: AnthropicMessagesRequest

    @model_validator(mode="after")
    def _params_not_stream(self) -> AnthropicBatchItem:
        if self.params.stream:
            raise ValueError("stream is not supported inside a message batch")
        return self


class AnthropicBatchCreate(_Model):
    """``POST /v1/messages/batches`` body — the requests ride inline (no
    input-file indirection like the OpenAI batch surface).``

    ``callback_url``/``callback_secret`` are the fx1 webhook extension —
    same terminal-delivery contract as ``/v1/batches`` (the finished
    ``message_batch`` envelope POSTs to the URL once, HMAC-signed when the
    secret is set; the secret never serializes onto the record)."""

    model_config = ConfigDict(extra="allow")

    requests: list[AnthropicBatchItem] = Field(min_length=1)
    # fx1 extension — terminal webhook (mirrors OpenAIBatchRequest)
    callback_url: str | None = None
    callback_secret: str | None = None

    @field_validator("callback_url")
    @classmethod
    def _callback_url_http(cls, v: str | None) -> str | None:
        return check_callback_url(v)

    @model_validator(mode="after")
    def _batch_valid(self) -> AnthropicBatchCreate:
        ids = [r.custom_id for r in self.requests]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"requests[].custom_id must be unique within the batch — duplicated: {dupes}"
            )
        if self.callback_secret is not None and not self.callback_url:
            raise ValueError("callback_secret requires callback_url")
        return self


class AnthropicBatchCounts(_Model):
    """``request_counts`` — Anthropic's tallies stay all-``processing``
    until the batch ends, then the terminal split lands at once."""

    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


def _rfc3339(ts: float | int | None) -> str | None:
    """Unix seconds → Anthropic's RFC 3339 ``...Z`` timestamp."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def anthropic_batch_object(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Internal batch record → the Anthropic ``message_batch`` object.

    ``results_url`` is null until ``processing_status`` is ``ended`` (the
    Anthropic contract — results exist only once every request has a
    terminal row). The ``callback_*`` keys are the fx1 webhook extension —
    the same verdict fields the OpenAI batch envelope carries."""
    ended = rec.get("status") == "ended"
    return {
        "id": rec["batch_id"],
        "type": "message_batch",
        "processing_status": rec["status"],
        "request_counts": dict(rec["request_counts"]),
        "ended_at": _rfc3339(rec.get("ended_at")),
        "created_at": _rfc3339(rec["created_at"]),
        "expires_at": _rfc3339(rec["expires_at"]),
        "cancel_initiated_at": _rfc3339(rec.get("cancel_initiated_at")),
        "archived_at": None,
        "results_url": (f"/v1/messages/batches/{rec['batch_id']}/results" if ended else None),
        "callback_url": rec.get("callback_url"),
        "callback_status": rec.get("callback_status"),
        "callback_attempts": rec.get("callback_attempts", 0),
        "callback_error": rec.get("callback_error"),
    }


def anthropic_batch_result(custom_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """One results-JSONL row — ``{custom_id, result}`` where ``result``
    is ``{type: 'succeeded', message}`` / ``{type: 'errored', error}`` /
    ``{type: 'canceled'}`` / ``{type: 'expired'}``."""
    return {"custom_id": custom_id, "result": result}
