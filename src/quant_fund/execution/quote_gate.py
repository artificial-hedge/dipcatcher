"""Fail closed when a quote is too old, skewed, duplicated, or out of order.

Callers must invoke this before they record an order or ask a broker to send
one. A halt means no order was sent.
"""

from __future__ import annotations

from typing import Any


class QuoteHalt(Exception):
    """Market data cannot be used. No order was sent."""


def _whole_ns(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _clock_problem(symbol: str, quote_ns: int, now_ns: int, max_age_ns: int) -> str | None:
    if not _whole_ns(max_age_ns) or not _whole_ns(quote_ns) or not _whole_ns(now_ns):
        return f"quote clock for {symbol} is not usable; no order sent"
    if quote_ns > now_ns:
        return f"clock skew for {symbol}: quote is ahead of the order clock; no order sent"
    if now_ns - quote_ns > max_age_ns:
        return f"stale quote for {symbol}: older than {max_age_ns} ns; no order sent"
    return None


def note_quote(
    *,
    symbol: str,
    quote_ns: int,
    now_ns: int,
    max_age_ns: int,
    previous_ns: int | None,
) -> None:
    """Record a new tick. Halt on skew, stale, out-of-order, or a duplicate timestamp."""
    problem = _clock_problem(symbol, quote_ns, now_ns, max_age_ns)
    if problem:
        raise QuoteHalt(problem)
    if previous_ns is not None and quote_ns < previous_ns:
        raise QuoteHalt(f"out-of-order quote for {symbol}; no order sent")
    if previous_ns is not None and quote_ns == previous_ns:
        raise QuoteHalt(f"duplicate quote for {symbol}; no order sent")


def require_fresh_quote(*, symbol: str, quote_ns: int, now_ns: int, max_age_ns: int) -> None:
    """Reuse an already-recorded quote. Halt on skew or age only (not a new tick)."""
    problem = _clock_problem(symbol, quote_ns, now_ns, max_age_ns)
    if problem:
        raise QuoteHalt(problem)


def require_monotonic_event_times(times: list[Any]) -> None:
    """Paper panel: equal timestamps are one cross-section. A step backward halts."""
    last: Any = None
    for raw in times:
        if last is not None and raw < last:
            raise ValueError("out-of-order market data; no order sent")
        if last is not None and raw == last:
            continue
        last = raw
