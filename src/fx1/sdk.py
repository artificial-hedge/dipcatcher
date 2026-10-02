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

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fx1 import __version__
from fx1.harness import Harness, HarnessCommand, HarnessResult, HarnessRole
from fx1.honesty import validate_fx1_output
from fx1.serve.backends import (
    BackendNotConfiguredError,
    InferenceBackend,
    StreamingBackend,
    get_backend,
)
from fx1.serve.chat import cited_complete
from quant_fund.research.receipt_v2 import verify_receipt_payload

__all__ = [
    "BackendNotConfiguredError",
    "CompletionResult",
    "Fx1Harness",
    "OpsMetrics",
    "ReceiptVerdict",
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


class Fx1Harness:
    """In-process harness client: registry, runs, gated complete, verify."""

    def __init__(
        self,
        harness: Harness | None = None,
        backend_resolver: BackendResolver | None = None,
    ) -> None:
        self._harness = harness or Harness()
        self._resolve_backend = backend_resolver or get_backend

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

    # ---- gated completion ----------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """One chat completion through the honesty gate.

        ``backend`` is one of ``hosted_k3`` / ``local_fx1`` / ``byok``.
        ``local_fx1`` requires ``checkpoint_dir`` here or
        ``FX1_CHECKPOINT_DIR`` in the environment; it may only be passed for
        ``local_fx1``. The backend is always closed afterwards — engines
        spawned by ``LocalFx1Backend`` never leak.
        """
        backend_obj = self._resolve_completion_backend(backend, checkpoint_dir, backend_kwargs)
        try:
            content = cited_complete(backend_obj, messages, receipt_hashes=receipt_hashes)
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        return CompletionResult(
            backend=backend,
            model=model_name if isinstance(model_name, str) else None,
            content=content,
            receipt_hashes=tuple(receipt_hashes or ()),
        )

    def complete_many(
        self,
        batch: list[list[dict[str, str]]],
        *,
        backend: str = "local_fx1",
        checkpoint_dir: str | Path | None = None,
        receipt_hashes: list[str] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        max_workers: int = 4,
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
        backend_obj = self._resolve_completion_backend(backend, checkpoint_dir, backend_kwargs)
        try:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(batch)), thread_name_prefix="fx1-complete"
            ) as pool:
                contents = list(
                    pool.map(
                        lambda msgs: cited_complete(
                            backend_obj, msgs, receipt_hashes=receipt_hashes
                        ),
                        batch,
                    )
                )
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()
        model_name = getattr(backend_obj, "_model", None)
        return [
            CompletionResult(
                backend=backend,
                model=model_name if isinstance(model_name, str) else None,
                content=content,
                receipt_hashes=tuple(receipt_hashes or ()),
            )
            for content in contents
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
    ) -> list[str]:
        """Token-delta chunks of one gated completion.

        Mirrors ``POST /harness/complete/stream``: the backend's SSE deltas
        are buffered, the joined text passes the honesty gate, and the
        chunk list is returned (with the evidence footer appended as a
        final chunk when ``receipt_hashes`` is given). Buffer-then-gate is
        the contract — an in-process consumer never holds ungated bytes
        either. A backend without ``stream`` raises ``NotImplementedError``
        (501-class); the backend is always closed afterwards.
        """
        backend_obj = self._resolve_completion_backend(backend, checkpoint_dir, backend_kwargs)
        try:
            if not isinstance(backend_obj, StreamingBackend):
                raise NotImplementedError(f"backend {backend!r} does not support streaming")
            chunks = list(backend_obj.stream(messages))
            joined = "".join(chunks)
            validate_fx1_output(joined)
            if receipt_hashes:
                chunks.append(
                    "\n\nEvidence: "
                    + ", ".join(f"`{h[:16]}…`" for h in receipt_hashes)
                    + " — verify with `dipcatcher verify-research`."
                )
            return chunks
        finally:
            closer = getattr(backend_obj, "close", None)
            if callable(closer):
                closer()

    def _resolve_completion_backend(
        self,
        backend: str,
        checkpoint_dir: str | Path | None,
        backend_kwargs: dict[str, Any] | None,
    ) -> Any:
        """Checkpoint contract + backend resolution shared by completes."""
        kwargs: dict[str, Any] = dict(backend_kwargs or {})
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
