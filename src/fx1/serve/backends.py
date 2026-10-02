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
    ``user``) pass through verbatim — the upstream decides whether each
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
    messages: list[dict[str, str]],
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


def _openai_chat_stream(
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
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


class InferenceBackend(Protocol):
    """Chat-completion interface shared by all fx-1 backends."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...


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
        messages: list[dict[str, str]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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

    def stream(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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

    def stream(
        self,
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
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

    def stream(
        self,
        messages: list[dict[str, str]],
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
