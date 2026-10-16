"""Thread-safe API accounting and its public metrics response models."""

from __future__ import annotations

import threading
import time
from typing import Any

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendCompletionStats(_Model):
    """Per-backend outcomes, cumulative latency, and reported token totals."""

    ok: int = 0
    error: int = 0
    latency_count: int = 0
    latency_sum_ms: float = 0.0
    latency_buckets: dict[str, int] = {}
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_calls: int = 0


class MetricsResponse(_Model):
    """Point-in-time ops snapshot: totals since process start."""

    uptime_s: float
    requests_total: int
    errors_total: int
    by_status: dict[str, int]
    inflight: int
    inflight_watermark: int
    max_inflight: int
    draining: bool
    rate_limited_total: int = 0
    complete: dict[str, BackendCompletionStats] = {}


_LAT_BUCKETS_MS = (100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0, 60000.0)


def _lat_bucket_labels(counts: list[int]) -> dict[str, int]:
    """Cumulative histogram buckets, including the final +Inf bucket."""
    labels = [f"{edge:g}" for edge in _LAT_BUCKETS_MS] + ["+Inf"]
    return dict(zip(labels, counts, strict=True))


class _Metrics:
    """Request counters, inflight gauge, and backend completion histograms."""

    def __init__(self, max_inflight: int) -> None:
        self.started = time.monotonic()
        self.max_inflight = max_inflight
        self._cond = threading.Condition()
        self._requests_total = 0
        self._errors_total = 0
        self._by_status: dict[int, int] = {}
        self._inflight = 0
        self._watermark = 0
        self._rate_limited_total = 0
        self._complete: dict[str, dict[str, Any]] = {}
        self.draining = threading.Event()

    def record_complete(
        self,
        backend: str,
        ok: bool,
        latency_ms: float,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Count only calls that reached a model and the usage it reported."""
        with self._cond:
            st = self._complete.setdefault(
                backend,
                {
                    "ok": 0,
                    "error": 0,
                    "count": 0,
                    "sum": 0.0,
                    "buckets": [0] * (len(_LAT_BUCKETS_MS) + 1),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "usage_calls": 0,
                },
            )
            st["ok" if ok else "error"] += 1
            st["count"] += 1
            st["sum"] += latency_ms
            for i, edge in enumerate(_LAT_BUCKETS_MS):
                if latency_ms <= edge:
                    st["buckets"][i] += 1
            st["buckets"][-1] += 1
            if usage:
                st["usage_calls"] += 1
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = usage.get(k)
                    if isinstance(v, int) and not isinstance(v, bool):
                        st[k] += v

    def record_rate_limited(self) -> None:
        with self._cond:
            self._rate_limited_total += 1

    def record(self, status: int) -> None:
        with self._cond:
            self._requests_total += 1
            self._by_status[status] = self._by_status.get(status, 0) + 1
            if status >= 400:
                self._errors_total += 1

    def acquire(self) -> None:
        with self._cond:
            self._inflight += 1
            self._watermark = max(self._watermark, self._inflight)

    def release(self) -> None:
        with self._cond:
            self._inflight -= 1
            self._cond.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait for all admitted requests to leave the pool."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def snapshot(self) -> MetricsResponse:
        with self._cond:
            return MetricsResponse(
                uptime_s=round(time.monotonic() - self.started, 3),
                requests_total=self._requests_total,
                errors_total=self._errors_total,
                by_status={str(k): v for k, v in sorted(self._by_status.items())},
                inflight=self._inflight,
                inflight_watermark=self._watermark,
                max_inflight=self.max_inflight,
                draining=self.draining.is_set(),
                rate_limited_total=self._rate_limited_total,
                complete={
                    b: BackendCompletionStats(
                        ok=st["ok"],
                        error=st["error"],
                        latency_count=st["count"],
                        latency_sum_ms=round(st["sum"], 3),
                        latency_buckets={**_lat_bucket_labels(st["buckets"])},
                        prompt_tokens=st["prompt_tokens"],
                        completion_tokens=st["completion_tokens"],
                        total_tokens=st["total_tokens"],
                        usage_calls=st["usage_calls"],
                    )
                    for b, st in sorted(self._complete.items())
                },
            )
