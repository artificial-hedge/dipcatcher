"""Deterministic idempotency keys for a live submit.

The caller key is used as given. Otherwise the key is a pure function of the
order, so a retry after a timeout or a restart addresses the same broker order.
A finished order does not occupy that key: the next distinct submit steps the
generation so a later flatten is not a replay of the fill.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

# Statuses that will not be retried. A live or in-flight status keeps the key.
_FINISHED = frozenset(
    {
        "filled",
        "canceled",
        "cancelled",
        "rejected",
        "expired",
        "denied",
        "not_acknowledged",
        "replaced",
        "done_for_day",
        "suspended",
    }
)


def deterministic_key(
    *,
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    client_order_id: str | None,
    generation: int = 0,
) -> str:
    """Key carried on every submit. Same order, same key."""
    if client_order_id:
        return str(client_order_id)
    payload = f"{symbol}|{side}|{quantity:.8f}|{limit_price:.8f}"
    if generation:
        payload = f"{payload}|{generation}"
    return "k" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


def assign_key(
    *,
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    client_order_id: str | None,
    known: Sequence[tuple[str, str]] = (),
) -> str:
    """Caller id, or the generated id of the live order, or the next generation."""
    if client_order_id:
        return str(client_order_id)
    by_status = {key: status for key, status in known}
    generation = 0
    while generation < 10_000:
        candidate = deterministic_key(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=None,
            generation=generation,
        )
        status = by_status.get(candidate)
        if status is None or status not in _FINISHED:
            return candidate
        generation += 1
    raise ValueError("idempotency generation exhausted")
