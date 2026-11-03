"""Per-order fat-finger checks. A breach is a reason, never a fill.

The collar is measured against the last valid mark. Equality with either
edge is inside the band. A missing or non-finite mark fails closed.
"""

from __future__ import annotations

import math


def fat_finger_reasons(
    *,
    quantity: float,
    limit_price: float,
    mark: float | None,
    max_quantity: float,
    max_notional: float,
    collar_bps: float,
) -> tuple[str, ...]:
    """Return the caps the order breaks. Empty means the order is inside them."""
    reasons: list[str] = []
    if not math.isfinite(quantity) or quantity <= 0.0 or quantity > max_quantity:
        reasons.append("max_order_quantity")
    if not math.isfinite(quantity) or not math.isfinite(limit_price):
        notional = float("nan")
    else:
        notional = quantity * limit_price
    if not math.isfinite(notional) or notional > max_notional:
        reasons.append("max_order_notional")
    if (
        mark is None
        or not math.isfinite(mark)
        or mark <= 0.0
        or not math.isfinite(collar_bps)
        or collar_bps < 0.0
    ):
        reasons.append("price_collar")
        return tuple(reasons)
    width = collar_bps / 10_000.0
    lower = mark * (1.0 - width)
    upper = mark * (1.0 + width)
    if not math.isfinite(limit_price) or limit_price < lower or limit_price > upper:
        reasons.append("price_collar")
    return tuple(reasons)
