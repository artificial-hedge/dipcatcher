"""harness bench — the sealed perf gate.

``fx1 harness bench`` times ``n`` gated ``complete`` calls at bounded
concurrency on either leg — the in-process ``Fx1Harness`` or a remote
``HarnessClient`` — and emits an ops-receipt-sealable record: the prompt
is ``prompt_sha256`` (never embedded), per-request latencies reduce to
the percentile card + throughput + honest error accounting, and every
parameter bounds-checks before a single token is spent. ``--receipt``
prints the sealed ``fx1_bench_result.v1`` doc, verifiable with
``dipcatcher verify-receipt`` / ``POST /receipts/verify`` like every
other evidence artifact.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

__all__ = ["DEFAULT_BENCH_PROMPT", "run_bench"]

DEFAULT_BENCH_PROMPT = "Summarize the Kelly criterion in one sentence."

_MAX_N = 4096
_MAX_CONCURRENCY = 256
_MAX_WARMUP = 256
_MAX_PROMPT_CHARS = 32768
_MAX_MAX_TOKENS = 262144


class _Completer(Protocol):
    """Structural type both legs satisfy: ``Fx1Harness.complete`` and
    ``HarnessClient.complete`` share the same call shape."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str,
        max_tokens: int | None,
        timeout_s: float | None,
        seed: int | None,
        byok: dict[str, str] | None,
    ) -> Any: ...


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' convention)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _check_int(name: str, value: int, lo: int, hi: int) -> None:
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


def run_bench(
    surface: _Completer,
    *,
    n: int = 32,
    concurrency: int = 4,
    warmup: int = 2,
    prompt: str = DEFAULT_BENCH_PROMPT,
    max_tokens: int = 32,
    timeout_s: float = 30.0,
    backend: str = "local_fx1",
    byok: dict[str, str] | None = None,
    seed: int | None = None,
    mode: str = "in_process",
) -> dict[str, Any]:
    """Time ``n`` gated completions at ``concurrency`` workers.

    ``warmup`` requests run sequentially first, unmeasured — they absorb
    engine spin-up so the measured window reflects steady state. Every
    measured request records its own monotonic latency; errors are
    counted by exception class, never swallowed. Token totals come from
    each result's ``usage`` where the surface reports it.
    """
    _check_int("n", n, 1, _MAX_N)
    _check_int("concurrency", concurrency, 1, _MAX_CONCURRENCY)
    _check_int("warmup", warmup, 0, _MAX_WARMUP)
    _check_int("max_tokens", max_tokens, 1, _MAX_MAX_TOKENS)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(f"prompt exceeds {_MAX_PROMPT_CHARS} chars")
    if mode not in ("in_process", "remote"):
        raise ValueError(f"mode must be in_process|remote, got {mode!r}")

    messages = [{"role": "user", "content": prompt}]

    def _once() -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            result = surface.complete(
                messages,
                backend=backend,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                seed=seed,
                byok=byok,
            )
        except Exception as exc:  # noqa: BLE001 — a bench records the fault class
            return {
                "ok": False,
                "latency_s": time.monotonic() - t0,
                "error": type(exc).__name__,
            }
        usage = result.usage if isinstance(getattr(result, "usage", None), dict) else {}
        return {
            "ok": True,
            "latency_s": time.monotonic() - t0,
            "error": None,
            "prompt_tokens": usage.get("prompt_tokens")
            if isinstance(usage.get("prompt_tokens"), int)
            else None,
            "completion_tokens": usage.get("completion_tokens")
            if isinstance(usage.get("completion_tokens"), int)
            else None,
            "model": getattr(result, "model", None),
            "backend": getattr(result, "backend", None),
        }

    for _ in range(warmup):
        _once()

    started_wall = time.time()
    started = time.monotonic()
    if concurrency == 1:
        samples = [_once() for _ in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            samples = list(pool.map(lambda _i: _once(), range(n)))
    wall_s = time.monotonic() - started

    lat = sorted(s["latency_s"] for s in samples)
    errors: dict[str, int] = {}
    prompt_tokens = 0
    completion_tokens = 0
    usage_reported = 0
    models: set[str] = set()
    backends: set[str] = set()
    for s in samples:
        if not s["ok"]:
            cls = str(s["error"])
            errors[cls] = errors.get(cls, 0) + 1
            continue
        if s["model"]:
            models.add(str(s["model"]))
        if s["backend"]:
            backends.add(str(s["backend"]))
        if s["prompt_tokens"] is not None or s["completion_tokens"] is not None:
            usage_reported += 1
            prompt_tokens += int(s["prompt_tokens"] or 0)
            completion_tokens += int(s["completion_tokens"] or 0)
    error_count = len(samples) - sum(1 for s in samples if s["ok"])

    record: dict[str, Any] = {
        "params": {
            "n": n,
            "concurrency": concurrency,
            "warmup": warmup,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "max_tokens": max_tokens,
            "timeout_s": timeout_s,
            "backend": backend,
            "seed": seed,
            "byok": byok is not None,
        },
        "mode": mode,
        "started_at_unix": started_wall,
        "metrics": {
            "measured_requests": n,
            "wall_s": wall_s,
            "requests_per_s": (n / wall_s) if wall_s > 0 else 0.0,
            "latency_s": {
                "min": lat[0] if lat else 0.0,
                "p50": _percentile(lat, 0.50),
                "p90": _percentile(lat, 0.90),
                "p95": _percentile(lat, 0.95),
                "p99": _percentile(lat, 0.99),
                "max": lat[-1] if lat else 0.0,
                "mean": statistics.fmean(lat) if lat else 0.0,
            },
            "error_count": error_count,
            "error_rate": error_count / n,
            "errors": dict(sorted(errors.items())),
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
            "completion_tokens_per_s": (completion_tokens / wall_s) if wall_s > 0 else 0.0,
            "usage_reported": usage_reported,
            "models": sorted(models),
            "backends": sorted(backends),
        },
    }
    return record
