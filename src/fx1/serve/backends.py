"""Inference backends for fx-1.

Two backends, one interface:
- ``HostedK3Backend`` — Moonshot's hosted K3 API. Serves as fx-1's teacher
  (tool-use trace generation) and as the day-one serving path.
- ``LocalFx1Backend`` — a fine-tuned fx-1 checkpoint. Fail-closed until a
  checkpoint with a valid model card exists on disk.

Credentials come from environment variables only — never hardcoded.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from fx1.modelcard import ModelCard

MOONSHOT_API_URL = "https://api.moonshot.ai/v1/chat/completions"

BYOK_BASE_URL_ENV = "FX1_BYOK_BASE_URL"
BYOK_API_KEY_ENV = "FX1_BYOK_API_KEY"
BYOK_MODEL_ENV = "FX1_BYOK_MODEL"

LOCAL_SERVE_URL_ENV = "FX1_LOCAL_SERVE_URL"
LOCAL_SERVE_CMD_ENV = "FX1_LOCAL_SERVE_CMD"
LOCAL_MODEL_ENV = "FX1_LOCAL_MODEL"
LOCAL_API_KEY_ENV = "FX1_LOCAL_API_KEY"
LOCAL_TIMEOUT_S_ENV = "FX1_LOCAL_TIMEOUT_S"
LOCAL_START_TIMEOUT_S_ENV = "FX1_LOCAL_START_TIMEOUT_S"


@dataclass(frozen=True)
class SamplingParams:
    """Declared per-call parameters for one completion.

    Only declared fields reach the wire beyond ``temperature`` — a
    provider that doesn't know ``seed`` never sees it. ``temperature``
    defaults to 0.0: eval/teacher runs stay deterministic unless the
    caller explicitly opts out, and the *resolved* set is what lands in
    the completion record as evidence of what was sampled.

    Decode knobs (``temperature``/``top_p``/``max_tokens``/``seed``/
    ``stop``/the penalty pair/``logit_bias``) shape generation; provider
    hints (``reasoning_effort``/``service_tier``/``prompt_cache_key``/
    ``prompt_cache_retention``/``verbosity``/``user``) pass through
    verbatim — the upstream decides whether each
    is meaningful; the harness records that they were requested.
    ``stop`` is additionally enforced harness-side
    (:func:`truncate_at_stops`) so providers that ignore it still ship
    the cut text.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    verbosity: str | None = None
    user: str | None = None

    def body_fields(self) -> dict[str, Any]:
        """The exact fields merged into the request body."""
        fields: dict[str, Any] = {
            "temperature": 0.0 if self.temperature is None else self.temperature
        }
        if self.top_p is not None:
            fields["top_p"] = self.top_p
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        if self.seed is not None:
            fields["seed"] = self.seed
        if self.stop is not None:
            fields["stop"] = list(self.stop)
        if self.presence_penalty is not None:
            fields["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            fields["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            fields["logit_bias"] = dict(self.logit_bias)
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            fields["service_tier"] = self.service_tier
        if self.prompt_cache_key is not None:
            fields["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention is not None:
            fields["prompt_cache_retention"] = self.prompt_cache_retention
        if self.verbosity is not None:
            fields["verbosity"] = self.verbosity
        if self.user is not None:
            fields["user"] = self.user
        return fields


def _stop_cut(text: str, stops: tuple[str, ...] | list[str] | None) -> int:
    """Index of the earliest stop-sequence occurrence, or len(text)."""
    cut = len(text)
    for s in stops or ():
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return cut


def truncate_at_stops(text: str, stops: tuple[str, ...] | list[str] | None) -> str:
    """Cut ``text`` before the earliest stop sequence — OpenAI ``stop``
    semantics, applied harness-side so every backend honors the contract
    even when the upstream doesn't (the matched sequence itself is
    excluded, per spec)."""
    return text[: _stop_cut(text, stops)]


def truncate_chunks(chunks: list[str], stops: tuple[str, ...] | list[str] | None) -> list[str]:
    """``truncate_at_stops`` over a delta stream — preserves the
    provider's chunk boundaries, emitting each chunk up to the cut and a
    slice of the straddling chunk (a stop that lands mid-delta truncates
    that delta rather than dropping whole chunks early)."""
    if not stops:
        return chunks
    cut = _stop_cut("".join(chunks), stops)
    out: list[str] = []
    pos = 0
    for chunk in chunks:
        if pos >= cut:
            break
        piece = chunk[: cut - pos]
        if piece:
            out.append(piece)
        pos += len(chunk)
    return out


class BackendNotConfiguredError(RuntimeError):
    """A backend whose required configuration is absent — a 503-class fault."""

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _chat_completions_url(base_url: str) -> str:
    """Normalize a BYOK base to the chat-completions route.

    Accepts ``https://host/v1``, ``https://host/v1/``, or an already-
    complete ``.../chat/completions`` URL.
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _openai_sibling_url(chat_url: str, route: str) -> str:
    """Derive a sibling route from a chat-completions URL.

    ``https://host/v1/chat/completions`` + ``"embeddings"`` →
    ``https://host/v1/embeddings``; a URL already ending in the route
    passes through unchanged.
    """
    base = chat_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith(f"/{route}"):
        return base
    return f"{base}/{route}"


def _env_float(name: str, override: float | None, default: float) -> float:
    """Resolve a seconds-valued knob: kwarg beats env beats default."""
    raw = override if override is not None else os.environ.get(name)
    if raw is None or raw == "":
        return default
    val = float(raw)
    if val <= 0:
        raise ValueError(f"{name} must be positive, got {val!r}")
    return val


def _extract_usage(payload: Any) -> dict[str, int] | None:
    """Pull the ``usage`` dict off an OpenAI-compatible response.

    Returns only int-valued keys (``prompt_tokens``/``completion_tokens``/
    ``total_tokens`` and friends); a missing or malformed block is ``None``
    — the caller reports no usage rather than fabricating counts."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    ints = {
        str(k): int(v)
        for k, v in usage.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return ints or None


class _UsageTracker:
    """Per-call + cumulative token accounting, shared by the three backends.

    ``last_usage`` is the most recent call's counts; ``total_usage``
    accumulates under a lock so a backend shared across a batch's worker
    threads reports correct totals. Backends whose endpoint omits usage
    keep ``last_usage=None`` and an empty ``total_usage`` forever."""

    def __init__(self) -> None:
        self.last_usage: dict[str, int] | None = None
        self.total_usage: dict[str, int] = {}
        self._usage_lock = threading.Lock()

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        self.last_usage = usage
        if not usage:
            return
        with self._usage_lock:
            for k, v in usage.items():
                self.total_usage[k] = self.total_usage.get(k, 0) + v


def _openai_chat_complete(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    api_key: str | None,
    label: str,
    sampling: SamplingParams | None = None,
) -> tuple[str, dict[str, int] | None]:
    """POST one OpenAI-compatible chat completion; map errors to RuntimeError.

    Returns ``(content, usage)`` — usage is ``None`` when the endpoint
    omits the block (older servers, local engines)."""
    # temperature pinned to 0 — eval/teacher runs must be deterministic;
    # unpinned sampling makes eval results unreproducible across replays.
    # temperature defaults to 0 — eval/teacher runs must be deterministic;
    # unpinned sampling makes eval results unreproducible across replays.
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            **(sampling or SamplingParams()).body_fields(),
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 — caller-declared endpoint  # nosec B310
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} endpoint {url} unreachable: {exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"malformed {label} completion payload: missing choices[0].message.content"
        ) from exc
    if not isinstance(content, str):
        raise RuntimeError(
            f"malformed {label} completion payload: content is {type(content).__name__}, not str"
        )
    return content, _extract_usage(payload)


def _tool_call_shape(raw: Any, label: str, i: int) -> dict[str, Any]:
    """Validate one ``tool_calls[]`` entry — fail closed on a malformed
    upstream frame rather than shipping a call fx-1 can't replay."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"malformed {label} tool_calls[{i}]: not an object")
    fn = raw.get("function")
    args = fn.get("arguments") if isinstance(fn, dict) else None
    if (
        not isinstance(raw.get("id"), str)
        or raw.get("type") != "function"
        or not isinstance(fn, dict)
        or not isinstance(fn.get("name"), str)
        or not isinstance(args, str)
    ):
        raise RuntimeError(
            f"malformed {label} tool_calls[{i}]: needs "
            "{id, type: 'function', function: {name, arguments}}"
        )
    return raw


def _openai_chat_complete_tools(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    parallel_tool_calls: bool | None,
    timeout_s: float,
    api_key: str | None,
    label: str,
    sampling: SamplingParams | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
) -> tuple[ToolCompletion, dict[str, int] | None]:
    """POST one OpenAI-compatible chat completion carrying ``tools``.

    ``tools``/``tool_choice``/``parallel_tool_calls`` pass through
    verbatim — the provider decides what each means. ``logprobs`` /
    ``top_logprobs`` pass through the same way: the response's
    ``choices[0].logprobs`` is echoed verbatim (``None`` when the
    provider stays silent — provider silence is its own answer, never
    fabricated). The response's ``choices[0]`` is validated fail-closed:
    ``content`` may be ``null`` (pure tool call), ``tool_calls`` entries
    must carry the OpenAI function-call shape, and a non-string
    ``content`` or malformed call raises ``RuntimeError`` — never a
    synthesized message.

    Returns ``(ToolCompletion, usage)`` — usage ``None`` when the
    endpoint omits the block."""
    body_map: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **(sampling or SamplingParams()).body_fields(),
    }
    if tools is not None:
        body_map["tools"] = tools
    if tool_choice is not None:
        body_map["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body_map["parallel_tool_calls"] = parallel_tool_calls
    if logprobs is not None:
        body_map["logprobs"] = logprobs
    if top_logprobs is not None:
        body_map["top_logprobs"] = top_logprobs
    body = json.dumps(body_map).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 — caller-declared endpoint  # nosec B310
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} endpoint {url} unreachable: {exc}") from exc
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"malformed {label} completion payload: missing choices[0].message"
        ) from exc
    if not isinstance(message, dict):
        raise RuntimeError(
            f"malformed {label} completion payload: message is {type(message).__name__}, not object"
        )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise RuntimeError(
            f"malformed {label} completion payload: content is "
            f"{type(content).__name__}, not str|null"
        )
    raw_calls = message.get("tool_calls")
    tool_calls: tuple[dict[str, Any], ...] | None = None
    if raw_calls is not None:
        if not isinstance(raw_calls, list):
            raise RuntimeError(
                f"malformed {label} completion payload: tool_calls is "
                f"{type(raw_calls).__name__}, not list"
            )
        tool_calls = tuple(_tool_call_shape(raw, label, i) for i, raw in enumerate(raw_calls))
    finish = choice.get("finish_reason")
    raw_lp = choice.get("logprobs")
    if raw_lp is not None and not isinstance(raw_lp, dict):
        raise RuntimeError(
            f"malformed {label} completion payload: logprobs is "
            f"{type(raw_lp).__name__}, not object|null"
        )
    return (
        ToolCompletion(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish if isinstance(finish, str) else None,
            logprobs=raw_lp,
        ),
        _extract_usage(payload),
    )


def _openai_chat_stream(
    url: str,
    *,
    model: str,
    messages: list[dict[str, Any]],
    timeout_s: float,
    api_key: str | None,
    label: str,
    usage_out: list[dict[str, int]] | None = None,
    sampling: SamplingParams | None = None,
) -> Iterator[str]:
    """POST one streaming OpenAI-compatible chat completion.

    Sends ``stream: true`` and yields each ``choices[0].delta.content``
    string as it arrives over server-sent events. Empty ``delta`` frames
    (role/handshake chunks) and ``data: [DONE]`` terminate the stream.
    Malformed frames fail closed as ``RuntimeError`` — a stream is never
    silently truncated.

    ``usage_out`` is an out-box: when any chunk carries an OpenAI-style
    ``usage`` dict (e.g. providers that attach per-chunk or final-frame
    usage), the last one lands in ``usage_out``. Never requested via
    ``stream_options`` — strict providers may reject unknown fields, so
    capture stays opportunistic.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            **(sampling or SamplingParams()).body_fields(),
        }
    ).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 — caller-declared endpoint  # nosec B310
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"malformed {label} stream chunk: not JSON ({data[:48]!r})"
                    ) from exc
                u = _extract_usage(chunk)
                if u is not None and usage_out is not None:
                    usage_out.clear()
                    usage_out.append(u)
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if choices is None:
                    if u is not None:
                        continue
                    raise RuntimeError(f"malformed {label} stream chunk: missing choices")
                if not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    raise RuntimeError(f"malformed {label} stream chunk: missing delta")
                content = delta.get("content")
                if content is None:
                    continue
                if not isinstance(content, str):
                    raise RuntimeError(
                        f"malformed {label} stream chunk: content is "
                        f"{type(content).__name__}, not str"
                    )
                yield content
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} endpoint {url} unreachable: {exc}") from exc


def _embedding_item_shape(raw: Any, label: str, i: int) -> dict[str, Any]:
    """Validate one ``data[]`` entry — fail closed on a malformed upstream
    frame rather than shipping a vector fx-1 can't attribute."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"malformed {label} embeddings payload: data[{i}] not an object")
    emb = raw.get("embedding")
    emb_ok = isinstance(emb, str) or (
        isinstance(emb, list)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in emb)
    )
    if (
        raw.get("object") != "embedding"
        or not isinstance(raw.get("index"), int)
        or isinstance(raw.get("index"), bool)
        or not emb_ok
    ):
        raise RuntimeError(
            f"malformed {label} embeddings payload: data[{i}] needs "
            "{object: 'embedding', index: int, embedding: number[]|base64}"
        )
    return raw


def _openai_embeddings_complete(
    url: str,
    *,
    model: str,
    input: Any,  # noqa: A002 — the wire field's own name
    encoding_format: str | None,
    dimensions: int | None,
    user: str | None,
    timeout_s: float,
    api_key: str | None,
    label: str,
) -> EmbeddingResult:
    """POST one OpenAI-compatible embeddings call; errors → RuntimeError.

    ``input`` forwards verbatim — strings or token arrays both ride the
    wire (the provider's tokenizer contract is its own). ``model`` is the
    request's model verbatim: embedding models name themselves, the
    link's pinned chat model does not apply. The response's ``data`` is
    validated fail-closed: each entry must carry the OpenAI embedding
    shape (``{object: "embedding", index: int, embedding:
    number[]|base64-str}``) — malformed frames raise, nothing is
    synthesized harness-side.
    """
    body_map: dict[str, Any] = {"model": model, "input": input}
    if encoding_format is not None:
        body_map["encoding_format"] = encoding_format
    if dimensions is not None:
        body_map["dimensions"] = dimensions
    if user is not None:
        body_map["user"] = user
    body = json.dumps(body_map).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 — caller-declared endpoint  # nosec B310
            payload = json.loads(response.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} endpoint {url} unreachable: {exc}") from exc
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(
            f"malformed {label} embeddings payload: data is {type(data).__name__}, not list"
        )
    return EmbeddingResult(
        data=tuple(_embedding_item_shape(raw, label, i) for i, raw in enumerate(data)),
        model=payload.get("model") if isinstance(payload.get("model"), str) else None,
        usage=_extract_usage(payload),
    )


@dataclass(frozen=True)
class EmbeddingResult:
    """A provider's ``/v1/embeddings`` answer.

    ``data`` is the verbatim ``data[]`` list (each validated to the
    OpenAI ``{object: "embedding", index, embedding}`` shape — number
    arrays or base64 strings as the provider sent them); ``model`` and
    ``usage`` echo the provider's own fields (``None`` under provider
    silence — never fabricated).
    """

    data: tuple[dict[str, Any], ...]
    model: str | None
    usage: dict[str, int] | None


@dataclass(frozen=True)
class ToolCompletion:
    """A provider's structured answer when the request carried ``tools``.

    ``content`` is the assistant's text (``None`` when the model went
    straight to a call — OpenAI emits ``content: null`` there);
    ``tool_calls`` is the verbatim ``choices[].message.tool_calls`` list
    (each ``{id, type: \"function\", function: {name, arguments}}``);
    ``finish_reason`` is the upstream's own reason (``tool_calls`` /
    ``stop`` / ``length`` / ...); ``logprobs`` is the verbatim
    ``choices[].logprobs`` payload when the request asked for it
    (``None`` when unrequested or the provider stayed silent). All four
    are provider-reported — nothing is synthesized harness-side.
    """

    content: str | None
    tool_calls: tuple[dict[str, Any], ...] | None
    finish_reason: str | None
    logprobs: dict[str, Any] | None = None


class InferenceBackend(Protocol):
    """Chat-completion interface shared by all fx-1 backends."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...


@runtime_checkable
class ToolBackend(Protocol):
    """Backends with a tool-calling channel.

    ``complete_with_tools`` is the optional second half of the contract:
    consumers capability-check via ``isinstance(b, ToolBackend)`` before
    routing a ``tools`` request, so a resolver-supplied backend without
    the channel fails closed (501) rather than silently dropping the
    caller's tool intent.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Backends with a ``/v1/embeddings`` channel.

    ``embeddings`` is the optional capability: consumers check
    ``isinstance(b, EmbeddingBackend)`` before routing an embeddings
    request, so a resolver-supplied backend without the channel fails
    closed (501) rather than fabricating vectors. ``input`` is the
    request's own value verbatim — string, string list, token array, or
    token-array list; ``model`` is the request's model verbatim
    (embedding models name themselves on the provider — a link's chat
    pin does not apply).
    """

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult: ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Backends that additionally emit token deltas over the SSE wire.

    ``stream`` is the optional second half of the contract: consumers check
    ``isinstance(b, StreamingBackend)`` before subscribing, so a resolver-
    supplied backend that can't stream fails closed (501) rather than
    faking chunking.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]: ...


class HostedK3Backend(_UsageTracker):
    """Hosted Kimi K3 via the Moonshot API (stdlib HTTP; no new deps)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k3",
        api_url: str = MOONSHOT_API_URL,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("MOONSHOT_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("MOONSHOT_API_KEY is not set; fx-1 never hardcodes credentials")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        super().__init__()
        self._model = model
        self._api_url = api_url
        self._timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        # temperature defaults to 0 — eval/teacher runs must be
        # deterministic; unpinned sampling makes eval results
        # unreproducible across replays.
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                **(sampling or SamplingParams()).body_fields(),
            }
        ).encode()
        request = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310 — pinned Moonshot API URL  # nosec B310
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError(
                f"malformed completion payload: content is {type(content).__name__}, not str"
            )
        self._record_usage(_extract_usage(payload))
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion — Moonshot's wire accepts the OpenAI
        ``tools`` and ``logprobs`` fields; they pass through verbatim."""
        result, usage = _openai_chat_complete_tools(
            self._api_url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas; Moonshot's API is OpenAI-SSE-compatible."""
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._api_url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
            usage_out=box,
            sampling=sampling,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        """``/v1/embeddings`` against the sibling route of the chat URL —
        the request's ``model`` reaches the wire verbatim (the chat pin
        ``kimi-k3`` is not an embedding model)."""
        result = _openai_embeddings_complete(
            _openai_sibling_url(self._api_url, "embeddings"),
            model=model,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="hosted_k3",
        )
        self._record_usage(result.usage)
        return result


class OpenAICompatBackend(_UsageTracker):
    """BYOK — any OpenAI-compatible chat-completions endpoint.

    Lets the whole fx-1 eval/bench fleet run against user-supplied
    providers (vLLM, SGLang, OpenRouter, Azure OpenAI, Ollama's compat
    shim, ...): every lane then emits the same sealed receipts and
    honesty gates regardless of whose weights sit behind the socket.

    Credentials come from ``FX1_BYOK_API_KEY`` (or the explicit kwarg)
    — never argv and never committed config. The endpoint is
    ``FX1_BYOK_BASE_URL`` + ``/chat/completions`` and the model name is
    ``FX1_BYOK_MODEL``. Explicit kwargs win over env. Fail-closed: a
    missing credential or a malformed payload raises, nothing is
    fabricated.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout_s: int = 120,
    ) -> None:
        url = base_url if base_url is not None else os.environ.get(BYOK_BASE_URL_ENV, "")
        key = api_key if api_key is not None else os.environ.get(BYOK_API_KEY_ENV, "")
        mdl = model if model is not None else os.environ.get(BYOK_MODEL_ENV, "")
        missing = [
            name
            for name, val in (
                (BYOK_BASE_URL_ENV, url),
                (BYOK_API_KEY_ENV, key),
                (BYOK_MODEL_ENV, mdl),
            )
            if not val
        ]
        if missing:
            raise BackendNotConfiguredError(
                "BYOK backend is not configured; set "
                + ", ".join(missing)
                + " (fx-1 never hardcodes credentials)"
            )
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise RuntimeError(f"{BYOK_BASE_URL_ENV} must be an http(s) URL, got {url!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        super().__init__()
        self._url = _chat_completions_url(url)
        self._api_key = key
        self._model = mdl
        self._timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        content, usage = _openai_chat_complete(
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            sampling=sampling,
        )
        self._record_usage(usage)
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion against the caller-declared endpoint —
        ``tools``/``tool_choice``/``parallel_tool_calls`` and
        ``logprobs``/``top_logprobs`` pass through verbatim; a provider
        that doesn't know them answers honestly (its own 4xx surfaces
        as ``RuntimeError``)."""
        result, usage = _openai_chat_complete_tools(
            self._url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas from the caller-declared endpoint."""
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
            usage_out=box,
            sampling=sampling,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        """``/v1/embeddings`` on the caller-declared endpoint — the
        request's ``model`` reaches the wire verbatim (the BYOK chat
        pin is not an embedding model); a provider that doesn't know
        the route answers honestly (its own 4xx surfaces)."""
        result = _openai_embeddings_complete(
            _openai_sibling_url(self._url, "embeddings"),
            model=model,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
            timeout_s=self._timeout_s,
            api_key=self._api_key,
            label="BYOK",
        )
        self._record_usage(result.usage)
        return result


class LocalFx1Backend(_UsageTracker):
    """A local fx-1 checkpoint served by a local OpenAI-compatible engine.

    fx-1 ships no in-process inference stack — the checkpoint is weights +
    card, and completion goes through an engine that serves those weights
    (vLLM, SGLang, llama.cpp's server, ...). Two ways to reach one:

    - ``FX1_LOCAL_SERVE_URL`` / ``serve_url=`` — attach to an already-running
      engine (e.g. ``http://127.0.0.1:8000/v1``).
    - ``FX1_LOCAL_SERVE_CMD`` / ``serve_cmd=`` — a spawn template for the
      engine, e.g. ``vllm serve \"$checkpoint_dir\" --port 8000``. The
      ``$checkpoint_dir`` placeholder expands to the *verified* checkpoint
      root and ``FX1_CHECKPOINT_DIR`` is exported to the child. The backend
      waits for the attach URL to answer, and ``close()`` terminates the
      child.

    ``serve_url`` is always required to complete — ``serve_cmd`` only decides
    who brings the engine up. Fail-closed: a missing card, a failed ship
    gate, an unsigned release under ``FX1_SIGNING_KEY``, or no engine config
    at all each raise instead of fabricating.

    Attestation enforcement: when ``require_signature=True`` (the default once
    FX1_SIGNING_KEY is set), unsigned or signature-mismatched checkpoints
    refuse to serve — the served model must be the model whose card passed
    the gates, verifiably.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        require_signature: bool | None = None,
        serve_url: str | None = None,
        serve_cmd: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        start_timeout_s: float | None = None,
    ) -> None:
        root = Path(checkpoint_dir)
        card_path = root / "modelcard.json"
        if not card_path.exists():
            raise FileNotFoundError(
                f"no model card at {card_path}; an fx-1 checkpoint without a card is not servable"
            )
        self.card = ModelCard.load(card_path)
        if not self.card.eval_delta.ship_eligible:
            raise RuntimeError(
                f"{self.card.version} failed the ship gate "
                "(honesty/domain/general deltas); refusing to serve"
            )
        if require_signature is None:
            require_signature = bool(os.environ.get("FX1_SIGNING_KEY"))
        if require_signature:
            from fx1.serve.signing import verify_release

            if not verify_release(root):
                raise RuntimeError(
                    "checkpoint release signature missing or invalid; "
                    "fx-1 serves only signed releases when FX1_SIGNING_KEY "
                    "is configured"
                )
        self._root = root
        url = serve_url if serve_url is not None else os.environ.get(LOCAL_SERVE_URL_ENV, "")
        if url:
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise RuntimeError(f"{LOCAL_SERVE_URL_ENV} must be an http(s) URL, got {url!r}")
        self._url = _chat_completions_url(url) if url else ""
        cmd = serve_cmd if serve_cmd is not None else os.environ.get(LOCAL_SERVE_CMD_ENV, "")
        self._serve_cmd = cmd
        self._model = (
            model if model is not None else (os.environ.get(LOCAL_MODEL_ENV) or self.card.version)
        )
        self._api_key = api_key if api_key is not None else os.environ.get(LOCAL_API_KEY_ENV, "")
        super().__init__()
        self._timeout_s = _env_float(LOCAL_TIMEOUT_S_ENV, timeout_s, 120.0)
        self._start_timeout_s = _env_float(LOCAL_START_TIMEOUT_S_ENV, start_timeout_s, 60.0)
        self._proc: subprocess.Popen[bytes] | None = None
        # Guards spawn/close so a shared backend is safe under complete_many.
        self._engine_lock = threading.Lock()

    def _engine_up(self) -> bool:
        """True once the attach URL's engine answers a models probe."""
        parsed = urllib.parse.urlparse(self._url)
        probe = f"{parsed.scheme}://{parsed.netloc}/v1/models"
        try:
            with urllib.request.urlopen(probe, timeout=1):  # noqa: S310 — declared local engine  # nosec B310
                return True
        except urllib.error.HTTPError:
            return True  # any HTTP response means a server is listening
        except (urllib.error.URLError, OSError):
            return False

    def _ensure_engine(self) -> None:
        """Spawn the engine from the serve template, once, and wait for it."""
        if not self._serve_cmd or self._proc is not None or self._engine_up():
            return
        with self._engine_lock:
            if self._proc is not None or self._engine_up():
                return  # another thread spawned/attached it while we waited
            argv = shlex.split(
                string.Template(self._serve_cmd).substitute(
                    checkpoint_dir=str(self._root), python=shlex.quote(sys.executable)
                )
            )
            env = dict(os.environ, FX1_CHECKPOINT_DIR=str(self._root))
            try:
                self._proc = subprocess.Popen(  # noqa: S603 — argv list, no shell  # nosec B603
                    argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except OSError as exc:
                raise RuntimeError(
                    f"failed to spawn local fx-1 engine {self._serve_cmd!r}: {exc}"
                ) from exc
            deadline = time.monotonic() + self._start_timeout_s
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        "local fx-1 engine exited during startup "
                        f"(rc={self._proc.returncode}): {self._serve_cmd!r}"
                    )
                if self._engine_up():
                    return
                time.sleep(0.1)
        self.close()
        raise RuntimeError(
            "local fx-1 engine did not become ready within "
            f"{self._start_timeout_s:g}s: {self._serve_cmd!r}"
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        if not self._url:
            raise BackendNotConfiguredError(
                "local_fx1 is not configured: set FX1_LOCAL_SERVE_URL to an "
                "OpenAI-compatible engine serving the checkpoint (fx-1 never "
                "hardcodes endpoints); FX1_LOCAL_SERVE_CMD may spawn one"
            )
        self._ensure_engine()
        content, usage = _openai_chat_complete(
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            sampling=sampling,
        )
        self._record_usage(usage)
        return content

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        """Structured completion — delegated to the serving engine. An
        engine/chat-template without tool or logprobs support answers its
        own error (honest 4xx → RuntimeError); nothing is faked
        harness-side."""
        if not self._url:
            raise BackendNotConfiguredError(
                "local_fx1 is not configured: set FX1_LOCAL_SERVE_URL to an "
                "OpenAI-compatible engine serving the checkpoint (fx-1 never "
                "hardcodes endpoints); FX1_LOCAL_SERVE_CMD may spawn one"
            )
        self._ensure_engine()
        result, usage = _openai_chat_complete_tools(
            self._url,
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            sampling=sampling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._record_usage(usage)
        return result

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Stream token deltas; the engine is ensured before subscribing."""
        self._ensure_engine()
        box: list[dict[str, int]] = []
        for tok in _openai_chat_stream(  # noqa: UP028 — trailer needs the box after exhaustion
            self._url,
            model=self._model,
            messages=messages,
            timeout_s=self._timeout_s,
            api_key=self._api_key or None,
            label="local_fx1",
            usage_out=box,
            sampling=sampling,
        ):
            yield tok
        if box:
            self._record_usage(box[-1])

    def close(self) -> None:
        """Terminate a spawned engine; a no-op when only attaching."""
        with self._engine_lock:
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self) -> LocalFx1Backend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def get_backend(kind: str, **kwargs: object) -> InferenceBackend:
    """Backend factory: ``hosted_k3``, ``local_fx1`` or ``byok``. Fail-closed."""
    backends = {
        "hosted_k3": HostedK3Backend,
        "local_fx1": LocalFx1Backend,
        "byok": OpenAICompatBackend,
    }
    if kind not in backends:
        raise KeyError(f"unknown backend {kind!r}; choose from {sorted(backends)}")
    return cast(InferenceBackend, backends[kind](**kwargs))
