"""Fat-finger caps. SYNTHETIC, derandomized, no sockets."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.live.fat_finger import fat_finger_reasons
from quant_fund.live.idempotency import deterministic_key

_FAST = settings(derandomize=True, max_examples=30, deadline=None)
_FINITE = dict(allow_nan=False, allow_infinity=False)


def _reasons(
    quantity: float,
    limit_price: float,
    mark: float | None,
    *,
    max_quantity: float = 100.0,
    max_notional: float = 1_000.0,
    collar_bps: float = 50.0,
) -> tuple[str, ...]:
    return fat_finger_reasons(
        quantity=quantity,
        limit_price=limit_price,
        mark=mark,
        max_quantity=max_quantity,
        max_notional=max_notional,
        collar_bps=collar_bps,
    )


@given(
    quantity=st.floats(min_value=100.0, max_value=1e6, exclude_min=True, **_FINITE),
    mark=st.floats(min_value=1.0, max_value=1_000.0, **_FINITE),
)
@_FAST
def test_quantity_above_the_cap_is_rejected(quantity: float, mark: float) -> None:
    reasons = _reasons(quantity, mark, mark, max_quantity=100.0, max_notional=1e18)
    assert "max_order_quantity" in reasons


@given(
    quantity=st.floats(min_value=0.01, max_value=100.0, **_FINITE),
    mark=st.floats(min_value=1.0, max_value=1_000.0, **_FINITE),
)
@_FAST
def test_quantity_inside_the_cap_is_not_a_quantity_breach(quantity: float, mark: float) -> None:
    reasons = _reasons(quantity, mark, mark, max_quantity=100.0, max_notional=1e18, collar_bps=0.0)
    assert "max_order_quantity" not in reasons
    assert "price_collar" not in reasons


@given(
    quantity=st.floats(min_value=1.0, max_value=10.0, **_FINITE),
    mark=st.floats(min_value=1.0, max_value=100.0, **_FINITE),
    multiple=st.floats(min_value=1.01, max_value=20.0, **_FINITE),
)
@_FAST
def test_notional_above_the_cap_is_rejected(quantity: float, mark: float, multiple: float) -> None:
    cap = quantity * mark
    reasons = _reasons(
        quantity,
        mark * multiple,
        mark * multiple,
        max_quantity=1e9,
        max_notional=cap,
        collar_bps=1e6,
    )
    assert "max_order_notional" in reasons


@given(
    mark=st.floats(min_value=10.0, max_value=500.0, **_FINITE),
    bps=st.floats(min_value=1.0, max_value=200.0, **_FINITE),
    beyond=st.floats(min_value=1.001, max_value=3.0, **_FINITE),
)
@_FAST
def test_price_outside_the_last_mark_is_rejected(mark: float, bps: float, beyond: float) -> None:
    width = bps / 10_000.0
    limit = mark * (1.0 + width) * beyond
    reasons = _reasons(1.0, limit, mark, max_quantity=1e9, max_notional=1e18, collar_bps=bps)
    assert "price_collar" in reasons


@given(
    mark=st.floats(min_value=10.0, max_value=500.0, **_FINITE),
    bps=st.floats(min_value=0.0, max_value=200.0, **_FINITE),
)
@_FAST
def test_price_on_the_collar_edge_is_inside(mark: float, bps: float) -> None:
    width = bps / 10_000.0
    limit = mark * (1.0 + width)
    reasons = _reasons(1.0, limit, mark, max_quantity=1e9, max_notional=1e18, collar_bps=bps)
    assert "price_collar" not in reasons


@given(
    quantity=st.floats(min_value=1.0, max_value=5.0, **_FINITE),
    limit_price=st.floats(min_value=1.0, max_value=50.0, **_FINITE),
)
@_FAST
def test_a_missing_mark_fails_closed(quantity: float, limit_price: float) -> None:
    assert "price_collar" in _reasons(quantity, limit_price, None, max_notional=1e18)


@given(
    symbol=st.sampled_from(("AAPL", "MSFT")),
    side=st.sampled_from(("buy", "sell")),
    quantity=st.floats(min_value=1.0, max_value=20.0, **_FINITE),
    limit_price=st.floats(min_value=1.0, max_value=200.0, **_FINITE),
)
@_FAST
def test_the_idempotency_key_is_stable_and_distinct(
    symbol: str, side: str, quantity: float, limit_price: float
) -> None:
    first = deterministic_key(
        symbol=symbol,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        client_order_id=None,
    )
    again = deterministic_key(
        symbol=symbol,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        client_order_id=None,
    )
    assert first == again
    assert first.startswith("k")
    assert math.isfinite(quantity)
    other = deterministic_key(
        symbol=symbol,
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        client_order_id="other-order",
    )
    assert other == "other-order"
    assert other != first
