"""Fx1Harness — the typed in-process client for the dipcatcher harness.

``fx1.serve.api`` exposes the harness over HTTP for remote fx-1 instances;
this module is the same surface for in-process callers — the SDK twin.
Same registry, same backend set, same honesty gate, same receipt verifier;
the only thing dropped is the socket (auth/body-cap are transport concerns,
not contract).

Call-shape parity with the API is the contract:

- ``commands(role=None)``   ↔ ``GET  /harness/commands``
- ``run(...)``              ↔ ``POST /harness/runs``
- ``complete(...)``         ↔ ``POST /harness/complete``
- ``verify_receipt(...)``   ↔ ``POST /receipts/verify``
- ``health()``              ↔ ``GET  /health``

Error taxonomy (the SDK raises; the API maps to status codes):

- ``KeyError``                    unknown command / backend        (404)
- ``ValueError``                  contract violation, escapes       (422)
- ``FileNotFoundError``           missing checkpoint                (422)
- ``BackendNotConfiguredError``   missing credentials/engine        (503)
- ``NotImplementedError``         backend lacks the operation       (501)
- ``Fx1HonestyError``             model output refused by the gate  (502)
- ``RuntimeError``                transport/other backend fault     (502)

Everything is injectable: pass a ``harness=`` or ``backend_resolver=``
double and no subprocess or network is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fx1 import __version__
from fx1.harness import Harness, HarnessCommand, HarnessResult, HarnessRole
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.serve.backends import (
    BackendNotConfiguredError,
    InferenceBackend,
    StreamingBackend,
    get_backend,
)
from fx1.serve.chat import cited_complete
from fx1.serve.receipt_store import SHA256_HEX, ReceiptIndex
from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload

__all__ = [
    "BackendNotConfiguredError",
    "CompletionRecord",
    "CompletionResult",
    "GateCheckResult",
    "ProbeResult",
    "Fx1Harness",
    "OpsMetrics",
    "ReceiptRef",
    "ReceiptVerdict",
    "StoredReceipt",
]

BackendResolver = Callable[..., InferenceBackend]


@dataclass(frozen=True)
class CompletionResult:
    """One gated completion — mirrors ``CompleteResponse`` on the API."""

    backend: str
    model: str | None
    content: str
    receipt_hashes: tuple[str, ...] = ()
    # Wire-only flag: True when the API replayed an idempotency-cached
    # response instead of re-running the model.
    replayed: bool = False
    # Endpoint-reported token counts for this call (None when the backend
    # has no usage channel). Never populated on ``complete_many`` items —
    # a shared backend can't attribute counts per prompt.
    usage: dict[str, int] | None = None
    # Handle into the surface's completion log — minted server-side on the
    # wire, by the SDK in-process. Every gated call is fetchable evidence.
    completion_id: str | None = None
    # Ordered fallback-chain trace — one entry per link tried (the last
    # is the serving link); empty when the primary served unchallenged.
    attempts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CompletionRecord:
    """One recorded gated call — mirrors ``CompletionRecord`` on the API.

    Hashes of the prompt and output, never the content: the log is
    evidence, not a transcript.
    """

    completion_id: str
    backend: str
    ok: bool
    latency_ms: float
    at: float
    prompt_sha256: str
    model: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None
    error_class: str | None = None
    output_sha256: str | None = None
    attempts: tuple[dict[str, Any], ...] | None = None


class _CompletionLog:
    """Bounded in-process ring of completion records, newest-first on
    read; the in-process twin of the API's log."""

    def __init__(self, cap: int = 256) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._items: dict[str, CompletionRecord] = {}

    def append(self, rec: CompletionRecord) -> None:
        with self._lock:
            self._items[rec.completion_id] = rec
            while len(self._items) > self._cap:
                self._items.pop(next(iter(self._items)))

    def get(self, completion_id: str) -> CompletionRecord | None:
        with self._lock:
            return self._items.get(completion_id)

    def latest(self, limit: int, backend: str | None) -> list[CompletionRecord]:
        with self._lock:
            items = sorted(self._items.values(), key=lambda r: r.at, reverse=True)
        if backend is not None:
            items = [r for r in items if r.backend == backend]
        return items[:limit]


_BACKEND_NAMES = ("hosted_k3", "local_fx1", "byok")


def _fallback_chain(backend: str, fallbacks: list[str] | None) -> list[str]:
    """Validated ordered chain: primary first, then distinct alternates —
    the same contract the API enforces on ``fallbacks``."""
    links = [backend, *(fallbacks or ())]
    if len(set(links)) != len(links) or len(links) > 3:
        raise ValueError("fallbacks must be distinct, at most two, and not repeat the primary")
    bad = [n for n in fallbacks or () if n not in _BACKEND_NAMES]
    if bad:
        raise ValueError(f"unknown fallback backends: {bad}")
    return links


def _check_link_kwargs(
    chain: list[str],
    checkpoint_dir: str | Path | None,
    byok: dict[str, str] | None,
) -> None:
    """Per-link kwargs must bind a chain link — the request-level twin of
    the API's ``_chain_valid`` (wire 422 ↔ sdk ValueError)."""
    if byok is not None and "byok" not in chain:
        raise ValueError("a byok override applies only to a 'byok' chain link")
    if checkpoint_dir is not None and "local_fx1" not in chain:
        raise ValueError("checkpoint_dir applies only to the local_fx1 backend")


def _messages_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GateCheckResult:
    """Honesty-gate verdict — mirrors ``GateCheckResponse`` on the API."""

    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    """Deep-health verdict for one backend — mirrors
    ``BackendProbeResponse`` on the API."""

    backend: str
    ok: bool
    model: str | None
    latency_ms: float
    error: str | None = None
    error_class: str | None = None


@dataclass(frozen=True)
class ReceiptVerdict:
    """Verifier output — mirrors ``ReceiptVerifyResponse`` on the API."""

    valid: bool
    path: str
    schema_tag: str
    kind: str | None
    verdict: str | None
    digest_convention: str | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReceiptRef:
    """One entry in the sealed-receipt store index — mirrors ReceiptIndexItem."""

    sha256: str
    name: str


@dataclass(frozen=True)
class StoredReceipt:
    """A fetched receipt: the verbatim sealed document plus the live
    verifier's verdict (``X-Fx1-Receipt-Valid`` on the wire, re-verify
    in-process here)."""

    sha256: str
    document: dict[str, Any]
    valid: bool


@dataclass(frozen=True)
class HarnessHealth:
    """Liveness + configured-backend presence booleans — no secret values."""

    status: str
    version: str
    registered_commands: int
    backends: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class OpsMetrics:
    """Remote ops snapshot — mirrors ``MetricsResponse`` on the API.

    Wire-only: the in-process surface has no HTTP layer to meter, so
    ``Fx1Harness`` deliberately does not expose this shape.
    """

    uptime_s: float
    requests_total: int
    errors_total: int
    by_status: dict[str, int] = field(default_factory=dict)
    inflight: int = 0
    inflight_watermark: int = 0
    max_inflight: int = 0
    draining: bool = False
    rate_limited_total: int = 0
    # per-backend outcome counters + cumulative latency buckets, keyed by
    # backend name (probe verdicts live under ``probe:<name>`` entries)
    complete: dict[str, dict[str, Any]] = field(default_factory=dict)


class Fx1Harness:
    """In-process harness client: registry, runs, gated complete, verify."""

    def __init__(
        self,
        harness: Harness | None = None,
        backend_resolver: BackendResolver | None = None,
        receipts_dir: str | Path = "receipts",
    ) -> None:
        self._harness = harness or Harness()
        self._resolve_backend = backend_resolver or get_backend
        self._receipts = ReceiptIndex(Path(receipts_dir))
        self._log = _CompletionLog()

    def _record_call(
        self,
        backend: str,
        model: str | None,
        ok: bool,
        latency_ms: float,
        usage: dict[str, int] | None,
        error: str | None,
        error_class: str | None,
        prompt_sha256: str,
        output_sha256: str | None,
        attempts: tuple[dict[str, Any], ...] | None = None,
    ) -> str:
        """Append one call to the completion log; returns its id."""
        cid = uuid.uuid4().hex
        self._log.append(
            CompletionRecord(
                completion_id=cid,
                backend=backend,
                model=model,
                ok=ok,
                latency_ms=latency_ms,
                at=time.time(),
                usage=usage,
                error=error,
                error_class=error_class,
                prompt_sha256=prompt_sha256,
                output_sha256=output_sha256,
                attempts=attempts,
            )
        )
        return cid

    def completions(self, *, limit: int = 50, backend: str | None = None) -> list[CompletionRecord]:
        """Newest-first window on the in-process completion log — the
        same audit evidence the API exposes at ``GET /harness/completions``."""
        return self._log.latest(limit, backend)

    def completion(self, completion_id: str) -> CompletionRecord:
        """Fetch one recorded call — KeyError when the id is unknown or
        already evicted (mirrors the wire's 404)."""
        rec = self._log.get(completion_id)
        if rec is None:
            raise KeyError(completion_id)
        return rec

    def completion_receipt(self, completion_id: str) -> dict[str, Any]:
        """Export one logged call as a sealed ``fx1_completion_record.v1``
        document — pass it to :meth:`verify_receipt` to check the seal."""
        from dataclasses import asdict  # noqa: PLC0415

        from fx1.serve.ops_receipt import (  # noqa: PLC0415
            completion_record_receipt,
        )

        rec = self._log.get(completion_id)
        if rec is None:
            raise KeyError(completion_id)
        return completion_record_receipt(asdict(rec))

    # ---- registry ------------------------------------------------------

    def commands(self, role: HarnessRole | None = None) -> list[str]:
        """Registered command names; anything unlisted is unreachable."""
        return [c.name for c in self._harness.list_commands(role)]

    def command(self, name: str) -> HarnessCommand:
        """The registered ``HarnessCommand`` (KeyError on unknown names)."""
        return self._harness.get(name)

    # ---- execution -----------------------------------------------------

    def run(
        self,
        command: str,
        extra_args: list[str] | None = None,
        *,
        config: Path | str | None = None,
    ) -> HarnessResult:
        """Execute a registered lab command (fail-closed on unknown names)."""
        return self._harness.run(
            command,
            extra_args,
            config=Path(config) if config is not None else None,
        )

    def run_receipt(self, result: HarnessResult) -> dict[str, Any]:
        """Seal an in-process run's outcome as ``fx1_run_result.v1`` —
        stdout/stderr digested, never content. The wire twin lives under
        ``GET /harness/jobs/{id}/receipt`` where the digested run record
        embeds in the job record."""
        from fx1.serve.ops_receipt import run_result_receipt  # noqa: PLC0415

        return run_result_receipt(result.model_dump())

    # ---- gated completion ----------------------------------------------

    def check_text(self, text: str) -> GateCheckResult:
        """Pre-flight text through the honesty gate in-process — never
        raises ``Fx1HonestyError``; the refusal rides ``ok=False, error``.
        Argument faults (non-str) propagate."""
        try:
            validate_fx1_output(text)
        except Fx1HonestyError as exc:
            return GateCheckResult(ok=False, error=str(exc))
        return GateCheckResult(ok=True)

    def probe_backend(
        self,
        backend: str,
        *,
        checkpoint_dir: str | Path | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        prompt: str = "ping",
    ) -> ProbeResult:
        """Deep health: run one minimal gated completion through the real
        resolver — same checkpoint/BYOK plumbing as ``complete``, so a
        BYOK probe tests the caller's own endpoint. Never raises for
        backend-side faults (that IS the verdict); argument faults raise
        as usual."""
        backend_obj = self._resolve_completion_backend(
            backend, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        t0 = time.monotonic()
        ok = False
        error: str | None = None
        error_class: str | None = None
        try:
            content = backend_obj.complete([{"role": "user", "content": prompt}])
            try:
                validate_fx1_output(content)
            except Fx1HonestyError as exc:
                error, error_class = str(exc), "honesty_refusal"
            else:
                ok = True
        except NotImplementedError as exc:
            error, error_class = str(exc), "NotImplementedError"
        except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
            error, error_class = str(exc), type(exc).__name__
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        return ProbeResult(
            backend=backend,
            ok=ok,
            model=model_name if isinstance(model_name, str) else None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            error=error,
            error_class=error_class,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
    ) -> CompletionResult:
        """One chat completion through the honesty gate.

        ``backend`` is one of ``hosted_k3`` / ``local_fx1`` / ``byok``.
        ``local_fx1`` requires ``checkpoint_dir`` here or
        ``FX1_CHECKPOINT_DIR`` in the environment; it may only be passed for
        ``local_fx1``. The backend is always closed afterwards — engines
        spawned by ``LocalFx1Backend`` never leak.

        ``fallbacks`` is an ordered chain of alternate backends tried after
        the primary — only on availability faults (unconfigured /
        transport). A gate refusal or a client error aborts the request;
        a refusal is a verdict, not a reason to spend another backend's
        capacity. Per-link kwargs: ``byok`` binds only a ``'byok'`` link,
        ``checkpoint_dir`` only a ``'local_fx1'`` link.
        """
        chain = _fallback_chain(backend, fallbacks)
        _check_link_kwargs(chain, checkpoint_dir, byok)
        prompt_sha256 = _messages_sha256(messages)
        attempts: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for cand in chain:
            t0 = time.monotonic()
            try:
                backend_obj = self._resolve_link(
                    cand, checkpoint_dir, backend_kwargs, byok, timeout_s
                )
            except (BackendNotConfiguredError, RuntimeError, ValueError) as exc:
                # resolve-level availability fault — same retriable class
                # the wire treats as backend_unavailable (503).
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                last_exc = exc
                continue
            except Exception as exc:
                # client errors abort — an unknown name, a bad receipt is
                # the request's fault, not the backend's.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                self._record_call(
                    cand,
                    None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                )
                raise
            t0 = time.monotonic()
            try:
                content = cited_complete(backend_obj, messages, receipt_hashes=receipt_hashes)
            except (BackendNotConfiguredError, RuntimeError) as exc:
                # availability fault — record the link, try the next.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                last_exc = exc
                closer = getattr(backend_obj, "close", None)
                if callable(closer):
                    closer()
                continue
            except Exception as exc:
                # refusals / capability gaps / client errors abort the chain.
                attempts.append(
                    {
                        "backend": cand,
                        "ok": False,
                        "error_class": type(exc).__name__,
                        "latency_ms": (time.monotonic() - t0) * 1000.0,
                    }
                )
                self._record_call(
                    cand,
                    getattr(backend_obj, "_model", None)
                    if isinstance(getattr(backend_obj, "_model", None), str)
                    else None,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    getattr(backend_obj, "last_usage", None)
                    if isinstance(getattr(backend_obj, "last_usage", None), dict)
                    else None,
                    str(exc),
                    type(exc).__name__,
                    prompt_sha256,
                    None,
                    tuple(attempts),
                )
                closer = getattr(backend_obj, "close", None)
                if callable(closer):
                    closer()
                raise
            model_name = getattr(backend_obj, "_model", None)
            usage = getattr(backend_obj, "last_usage", None)
            attempts.append(
                {
                    "backend": cand,
                    "ok": True,
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                }
            )
            cid = self._record_call(
                cand,
                model_name if isinstance(model_name, str) else None,
                True,
                (time.monotonic() - t0) * 1000.0,
                usage if isinstance(usage, dict) else None,
                None,
                None,
                prompt_sha256,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                tuple(attempts) if len(attempts) > 1 else None,
            )
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
            return CompletionResult(
                backend=cand,
                model=model_name if isinstance(model_name, str) else None,
                content=content,
                usage=usage if isinstance(usage, dict) else None,
                receipt_hashes=tuple(receipt_hashes or ()),
                completion_id=cid,
                attempts=tuple(attempts) if len(attempts) > 1 else (),
            )
        assert last_exc is not None  # every link failed retriably
        self._record_call(
            backend,
            None,
            False,
            0.0,
            None,
            str(last_exc),
            type(last_exc).__name__,
            prompt_sha256,
            None,
            tuple(attempts) if len(attempts) > 1 else None,
        )
        raise last_exc

    def complete_many(
        self,
        batch: list[list[dict[str, str]]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        max_workers: int = 4,
        fallbacks: list[str] | None = None,
    ) -> list[CompletionResult]:
        """Many gated completions over ONE shared backend instance.

        The backend resolves once and is shared across worker threads — a
        spawned local engine serves the whole batch, not one spawn per
        item (``LocalFx1Backend``'s spawn path is lock-guarded). Results
        come back in submission order; the lowest-index failure propagates
        after the pool drains. The backend is always closed afterwards.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if not batch:
            return []
        serving, backend_obj = self._resolve_chain(
            backend, fallbacks, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        model_name = getattr(backend_obj, "_model", None)
        model_str = model_name if isinstance(model_name, str) else None

        def _one(msgs: list[dict[str, str]]) -> tuple[str, str]:
            t0 = time.monotonic()
            p_sha = _messages_sha256(msgs)
            try:
                content = cited_complete(backend_obj, msgs, receipt_hashes=receipt_hashes)
            except Exception as exc:
                self._record_call(
                    serving,
                    model_str,
                    False,
                    (time.monotonic() - t0) * 1000.0,
                    None,
                    str(exc),
                    type(exc).__name__,
                    p_sha,
                    None,
                )
                raise
            cid = self._record_call(
                serving,
                model_str,
                True,
                (time.monotonic() - t0) * 1000.0,
                None,
                None,
                None,
                p_sha,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            return content, cid

        try:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(batch)), thread_name_prefix="fx1-complete"
            ) as pool:
                done = list(pool.map(_one, batch))
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        return [
            CompletionResult(
                backend=serving,
                model=model_str,
                content=content,
                receipt_hashes=tuple(receipt_hashes or ()),
                completion_id=cid,
            )
            for content, cid in done
        ]

    # ---- receipts --------------------------------------------------------

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        byok: dict[str, str] | None = None,
        timeout_s: float | None = None,
        fallbacks: list[str] | None = None,
    ) -> list[str]:
        """Token-delta chunks of one gated completion.

        Mirrors ``POST /harness/complete/stream``: the backend's SSE deltas
        are buffered, the joined text passes the honesty gate, and the
        chunk list is returned (with the evidence footer appended as a
        final chunk when ``receipt_hashes`` is given). Buffer-then-gate is
        the contract — an in-process consumer never holds ungated bytes
        either. A backend without ``stream`` raises ``NotImplementedError``
        (501-class); the backend is always closed afterwards.
        ``fallbacks`` applies at resolve level only — once a link is
        streaming there is no honest restart point.
        """
        serving, backend_obj = self._resolve_chain(
            backend, fallbacks, checkpoint_dir, backend_kwargs, byok, timeout_s
        )
        t0 = time.monotonic()
        prompt_sha256 = _messages_sha256(messages)
        try:
            if not isinstance(backend_obj, StreamingBackend):
                raise NotImplementedError(f"backend {serving!r} does not support streaming")
            chunks = list(backend_obj.stream(messages))
            joined = "".join(chunks)
            validate_fx1_output(joined)
            if receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
        except Exception as exc:
            self._record_call(
                serving,
                getattr(backend_obj, "_model", None)
                if isinstance(getattr(backend_obj, "_model", None), str)
                else None,
                False,
                (time.monotonic() - t0) * 1000.0,
                getattr(backend_obj, "last_usage", None)
                if isinstance(getattr(backend_obj, "last_usage", None), dict)
                else None,
                str(exc),
                type(exc).__name__,
                prompt_sha256,
                None,
            )
            raise
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        usage = getattr(backend_obj, "last_usage", None)
        self._record_call(
            serving,
            model_name if isinstance(model_name, str) else None,
            True,
            (time.monotonic() - t0) * 1000.0,
            usage if isinstance(usage, dict) else None,
            None,
            None,
            prompt_sha256,
            hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        )
        return chunks

    def _resolve_chain(
        self,
        backend: str,
        fallbacks: list[str] | None,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> tuple[str, Any]:
        """First resolvable link serves — resolve-level fallback for the
        shared-backend calls (batch / stream), same contract as the wire."""
        chain = _fallback_chain(backend, fallbacks)
        _check_link_kwargs(chain, checkpoint_dir, byok)
        last: Exception | None = None
        for cand in chain:
            try:
                return cand, self._resolve_link(
                    cand, checkpoint_dir, backend_kwargs, byok, timeout_s
                )
            except (BackendNotConfiguredError, RuntimeError) as exc:
                last = exc
        assert last is not None  # every link failed retriably
        raise last

    def _resolve_link(
        self,
        link: str,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> Any:
        """Resolve one fallback-chain link — per-link kwargs: ``byok``
        binds only a ``'byok'`` link, ``checkpoint_dir`` only ``'local_fx1'``."""
        return self._resolve_completion_backend(
            link,
            checkpoint_dir if link == "local_fx1" else None,
            backend_kwargs,
            byok if link == "byok" else None,
            timeout_s,
        )

    def _resolve_completion_backend(
        self,
        backend: str,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
        byok: dict[str, str] | None,
        timeout_s: float | None,
    ) -> Any:
        """Checkpoint contract + backend resolution shared by completes."""
        kwargs: dict[str, Any] = dict(backend_kwargs or {})
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        if byok is not None:
            if backend != "byok":
                raise ValueError("a byok override applies only to backend='byok'")
            required = {"base_url", "api_key", "model"}
            if set(byok) != required:
                raise ValueError(f"byok needs exactly {sorted(required)}, got {sorted(byok)}")
            parsed = urllib.parse.urlparse(byok["base_url"])
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"byok.base_url must be an http(s) URL, got {byok['base_url']!r}")
            kwargs.update(byok)
        if backend == "local_fx1":
            checkpoint = checkpoint_dir or os.environ.get("FX1_CHECKPOINT_DIR")
            if not checkpoint:
                raise ValueError(
                    "local_fx1 needs a checkpoint_dir argument or "
                    "FX1_CHECKPOINT_DIR in the environment"
                )
            kwargs["checkpoint_dir"] = str(checkpoint)
        elif checkpoint_dir is not None:
            raise ValueError("checkpoint_dir applies only to the local_fx1 backend")
        return self._resolve_backend(backend, **kwargs)

    def verify_receipt(self, receipt: dict[str, Any]) -> ReceiptVerdict:
        """Deep-verify a receipt object against the v2 contract."""
        result = verify_receipt_payload(receipt, path=Path("<sdk>"))
        return ReceiptVerdict(
            valid=result["valid"],
            path=result["path"],
            schema_tag=result["schema"],
            kind=result["kind"] if isinstance(result["kind"], str) else None,
            verdict=result["verdict"] if isinstance(result["verdict"], str) else None,
            digest_convention=result["digest_convention"],
            errors=tuple(result["errors"]),
            warnings=tuple(result["warnings"]),
        )

    def receipts(self) -> tuple[ReceiptRef, ...]:
        """Index the local sealed-receipt store — the in-process twin of
        ``GET /receipts``. An absent store raises FileNotFoundError."""
        if not self._receipts.available():
            raise FileNotFoundError(f"receipts store unavailable at {self._receipts.root}")
        return tuple(ReceiptRef(sha256=sha, name=path.name) for sha, path in self._receipts.items())

    def receipt(self, sha256: str) -> StoredReceipt:
        """Fetch one sealed receipt by content hash — the in-process twin of
        ``GET /receipts/{sha256}``: verbatim document plus a live re-verify.
        Unknown hashes raise KeyError; a malformed digest raises ValueError."""
        if SHA256_HEX.fullmatch(sha256) is None:
            raise ValueError(f"malformed receipt sha256: {sha256!r}")
        path = self._receipts.lookup(sha256)
        if path is None or not path.is_file():
            raise KeyError(f"receipt not found: {sha256}")
        return StoredReceipt(
            sha256=sha256,
            document=json.loads(path.read_bytes()),
            valid=bool(verify_receipt_file(path)["valid"]),
        )

    def verify_receipts(self, receipts: list[dict[str, Any]]) -> tuple[ReceiptVerdict, ...]:
        """Batch counterpart: in-process it's a loop; the wire client's twin
        is one POST — same return shape either way."""
        return tuple(self.verify_receipt(r) for r in receipts)

    # ---- health ----------------------------------------------------------

    def health(self) -> HarnessHealth:
        """Configured-backend presence flags — booleans only, never values."""
        from fx1.serve.backends import (
            BYOK_API_KEY_ENV,
            BYOK_BASE_URL_ENV,
            BYOK_MODEL_ENV,
            LOCAL_SERVE_CMD_ENV,
            LOCAL_SERVE_URL_ENV,
        )

        checkpoint_env = os.environ.get("FX1_CHECKPOINT_DIR", "")
        return HarnessHealth(
            status="ok",
            version=__version__,
            registered_commands=len(self._harness.list_commands()),
            backends={
                "hosted_k3": bool(os.environ.get("MOONSHOT_API_KEY")),
                "byok": all(
                    os.environ.get(n) for n in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)
                ),
                "local_fx1": bool(checkpoint_env)
                and (Path(checkpoint_env) / "modelcard.json").is_file()
                and bool(
                    os.environ.get(LOCAL_SERVE_URL_ENV) or os.environ.get(LOCAL_SERVE_CMD_ENV)
                ),
            },
        )
