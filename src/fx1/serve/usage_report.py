"""Usage accounting over the completion log — the billing/ops view.

Every gated model call lands one ``CompletionRecord`` on the bounded
ring (hashes only, never content). This module folds the visible
records into per-backend and per-model counters — requests, verdicts,
reported token usage, mean latency — on the API (``GET /harness/usage``)
and in-process on the SDK (``Fx1Harness.usage``). Same aggregate on both
legs, same model.

Honesty: the ring is bounded, so ``records_dropped`` (ring evictions)
and ``ring_cap`` travel with the report — a truncated window is declared,
never silently clipped. Token totals only count calls whose backend
reported usage (``usage_reported``); ``other_usage`` sums every
non-canonical int key verbatim so provider-specific counters aren't
lost. The report is accounting over *recorded* calls — it is not a
monetary claim.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

__all__ = [
    "UsageBucket",
    "UsageReport",
    "aggregate_usage",
]

_CANONICAL_USAGE_KEYS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})


class _Record(Protocol):
    """The fields both CompletionRecord shapes carry (API pydantic model
    and the SDK's frozen dataclass). Read-only members so a frozen
    dataclass conforms."""

    @property
    def backend(self) -> str: ...
    @property
    def model(self) -> str | None: ...
    @property
    def ok(self) -> bool: ...
    @property
    def latency_ms(self) -> float: ...
    @property
    def at(self) -> float: ...
    @property
    def usage(self) -> dict[str, int] | None: ...


class UsageBucket(BaseModel):
    """Aggregate counters for one slice of the completion log."""

    model_config = ConfigDict(extra="forbid")

    requests: int
    ok: int
    errors: int
    usage_reported: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    other_usage: dict[str, int]
    mean_latency_ms: float | None


class UsageReport(BaseModel):
    """One usage aggregate: the applied filters, ring accounting, the
    totals, and the per-backend/per-model splits."""

    model_config = ConfigDict(extra="forbid")

    generated_at: float
    since: float | None
    until: float | None
    backend: str | None
    model: str | None
    records_seen: int
    records_dropped: int
    ring_cap: int
    totals: UsageBucket
    by_backend: dict[str, UsageBucket]
    by_model: dict[str, UsageBucket]


def _bucket(records: list[_Record]) -> UsageBucket:
    requests = len(records)
    ok = sum(1 for r in records if r.ok)
    usage_reported = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    other: dict[str, int] = {}
    for rec in records:
        if rec.usage is None:
            continue
        usage_reported += 1
        for key, value in rec.usage.items():
            if key == "prompt_tokens":
                prompt_tokens += value
            elif key == "completion_tokens":
                completion_tokens += value
            elif key == "total_tokens":
                total_tokens += value
            elif isinstance(value, int):
                other[key] = other.get(key, 0) + value
    return UsageBucket(
        requests=requests,
        ok=ok,
        errors=requests - ok,
        usage_reported=usage_reported,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        other_usage=other,
        mean_latency_ms=(sum(r.latency_ms for r in records) / requests) if requests else None,
    )


def aggregate_usage(
    records: Sequence[_Record],
    *,
    cap: int,
    dropped: int,
    backend: str | None = None,
    model: str | None = None,
    since: float | None = None,
    until: float | None = None,
) -> UsageReport:
    """Fold ``records`` (already backend-filtered by the caller's log
    read, further narrowed here) into a usage report. `dropped` is the
    ring's eviction count — reported so a clipped window is explicit."""
    seen = [
        r
        for r in records
        if (model is None or r.model == model)
        and (since is None or r.at >= since)
        and (until is None or r.at <= until)
    ]
    by_backend: dict[str, list[_Record]] = {}
    by_model: dict[str, list[_Record]] = {}
    for rec in seen:
        by_backend.setdefault(rec.backend, []).append(rec)
        by_model.setdefault(rec.model if rec.model is not None else "(none)", []).append(rec)
    return UsageReport(
        generated_at=time.time(),
        since=since,
        until=until,
        backend=backend,
        model=model,
        records_seen=len(seen),
        records_dropped=dropped,
        ring_cap=cap,
        totals=_bucket(seen),
        by_backend={k: _bucket(v) for k, v in sorted(by_backend.items())},
        by_model={k: _bucket(v) for k, v in sorted(by_model.items())},
    )
