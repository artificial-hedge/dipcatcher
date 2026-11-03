"""Order-lifecycle comparisons at EPS.

A stable fill exactly one EPS ahead of the volatile fill is not a lead.
A fill of exactly EPS is not a quantity. A resting fill exactly one EPS
short of the order closes it. These rows check the specification, not a venue.
"""

from __future__ import annotations

from quant_fund.formal.order_lifecycle import EPS, OrderState, check_trace, invariant_violations


def test_stable_fill_equal_to_volatile_plus_eps_is_not_ahead() -> None:
    filled = 1.0
    order = OrderState(
        status="partial",
        qty=2.0,
        filled=filled,
        fills={"f": filled},
        s_status="partial",
        s_qty=2.0,
        s_filled=filled + EPS,
        s_fills={"f": filled},
    )
    assert invariant_violations(order) == []


def test_fill_of_exactly_eps_is_rejected() -> None:
    result = check_trace(
        [
            {"op": "submit", "order_id": "o", "qty": 1.0},
            {"op": "fill", "order_id": "o", "fill_id": "f", "qty": EPS},
        ]
    )
    assert result.violations
    assert "would exceed authorized" in result.violations[0]
    assert "f" not in result.book.orders["o"].fills


def test_resting_fill_exactly_one_eps_short_closes() -> None:
    result = check_trace(
        [
            {"op": "submit", "order_id": "o", "qty": 1.0},
            {
                "op": "fill",
                "order_id": "o",
                "fill_id": "f",
                "qty": 1.0 - EPS,
                "rest": True,
            },
        ]
    )
    assert result.ok
    assert result.book.orders["o"].status == "filled"
