"""Fail-closed transitions of the executable order-lifecycle specification.

The profitable-path enumerations live in ``tests/formal``; these unit tests pin
the *disabled* events: every guard that must raise ``LifecycleViolation`` and
every invariant that must be reported.
"""

from __future__ import annotations

import pytest

from quant_fund.formal.order_lifecycle import (
    Book,
    LifecycleViolation,
    OrderState,
    check_trace,
    enumerate_safety,
    invariant_violations,
    math_positive,
    step,
)


def _submitted(qty: float = 3.0, order_id: str = "A") -> Book:
    return step(Book(), {"op": "submit", "order_id": order_id, "qty": qty})


def _filled(qty: float = 3.0, order_id: str = "A") -> Book:
    book = _submitted(qty, order_id)
    return step(
        book, {"op": "fill", "order_id": order_id, "fill_id": "f1", "qty": qty, "rest": False}
    )


# -- submit --


def test_submit_is_disabled_on_a_working_order() -> None:
    book = _submitted()
    with pytest.raises(LifecycleViolation, match="submit from new"):
        step(book, {"op": "submit", "order_id": "A", "qty": 1.0})


@pytest.mark.parametrize("qty", [0.0, -1.0, 1e-9])
def test_submit_requires_a_positive_quantity(qty: float) -> None:
    with pytest.raises(LifecycleViolation, match="quantity must be positive"):
        step(Book(), {"op": "submit", "order_id": "A", "qty": qty})


def test_submit_is_disabled_after_termination() -> None:
    book = _filled()
    with pytest.raises(LifecycleViolation, match="submit from filled"):
        step(book, {"op": "submit", "order_id": "A", "qty": 3.0})
    assert book.orders["A"].filled == 3.0


def test_math_positive_boundary() -> None:
    assert math_positive(1e-8) is False
    assert math_positive(2e-8) is True


# -- reject / cancel / expire --


@pytest.mark.parametrize("op", ["reject", "cancel", "expire"])
def test_terminal_actions_require_a_working_order(op: str) -> None:
    with pytest.raises(LifecycleViolation, match=f"{op} from absent"):
        step(Book(), {"op": op, "order_id": "A"})
    with pytest.raises(LifecycleViolation, match=f"{op} from filled"):
        step(_filled(), {"op": op, "order_id": "A"})


@pytest.mark.parametrize("op", ["reject", "cancel", "expire"])
def test_terminal_actions_set_the_matching_status(op: str) -> None:
    book = step(_submitted(), {"op": op, "order_id": "A"})
    assert (
        book.orders["A"].status
        == {"reject": "rejected", "cancel": "canceled", "expire": "expired"}[op]
    )
    assert not invariant_violations(book.orders["A"])


# -- amend --


def test_amend_is_disabled_without_a_working_order() -> None:
    with pytest.raises(LifecycleViolation, match="amend from absent"):
        step(Book(), {"op": "amend", "order_id": "A", "qty": 5.0})
    with pytest.raises(LifecycleViolation, match="amend from canceled"):
        step(
            step(_submitted(), {"op": "cancel", "order_id": "A"}),
            {"op": "amend", "order_id": "A", "qty": 5.0},
        )


@pytest.mark.parametrize("qty", [3.0, 0.0, -2.0, 5e-9])
def test_amend_must_increase_above_filled(qty: float) -> None:
    with pytest.raises(LifecycleViolation, match="is not enabled"):
        step(_submitted(), {"op": "amend", "order_id": "A", "qty": qty})


def test_amend_after_partial_fill_keeps_partial_status() -> None:
    book = step(
        _submitted(4.0),
        {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 1.0, "rest": True},
    )
    assert book.orders["A"].status == "partial"
    amended = step(book, {"op": "amend", "order_id": "A", "qty": 6.0})
    assert amended.orders["A"].qty == 6.0
    assert amended.orders["A"].status == "partial"
    assert not invariant_violations(amended.orders["A"])


def test_amend_below_filled_is_refused() -> None:
    book = step(
        _submitted(4.0),
        {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 2.0, "rest": True},
    )
    with pytest.raises(LifecycleViolation, match="is not enabled"):
        step(book, {"op": "amend", "order_id": "A", "qty": 2.0})


# -- fills --


def test_fill_is_disabled_on_an_absent_order() -> None:
    with pytest.raises(LifecycleViolation, match="fill on absent"):
        step(Book(), {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 1.0})


@pytest.mark.parametrize("qty", [0.0, -1.0, 1e-9])
def test_fill_requires_a_positive_quantity(qty: float) -> None:
    with pytest.raises(LifecycleViolation, match="would exceed authorized"):
        step(_submitted(), {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": qty})


def test_overfill_is_refused_and_leaves_state_untouched() -> None:
    book = _submitted(3.0)
    with pytest.raises(LifecycleViolation, match="would exceed authorized 3.0"):
        step(book, {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 3.5})
    assert book.orders["A"].filled == 0.0


def test_fill_after_termination_is_refused() -> None:
    book = step(_submitted(), {"op": "cancel", "order_id": "A"})
    with pytest.raises(LifecycleViolation, match="fill on canceled"):
        step(book, {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 1.0})


def test_replayed_fill_id_is_a_stutter() -> None:
    book = _filled()
    replayed = step(book, {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 99.0})
    assert replayed.orders["A"].filled == 3.0
    assert replayed.orders["A"].fills == {"f1": 3.0}


def test_duplicate_fill_of_unknown_id_is_refused() -> None:
    with pytest.raises(LifecycleViolation, match="duplicate of unknown fill nope"):
        step(_submitted(), {"op": "duplicate_fill", "order_id": "A", "fill_id": "nope"})


def test_duplicate_fill_of_known_id_is_a_stutter() -> None:
    book = _filled()
    assert (
        step(book, {"op": "duplicate_fill", "order_id": "A", "fill_id": "f1"}).orders["A"].filled
        == 3.0
    )


def test_late_fill_is_disabled_while_working() -> None:
    with pytest.raises(LifecycleViolation, match="late_fill on new"):
        step(_submitted(), {"op": "late_fill", "order_id": "A", "fill_id": "f9"})


@pytest.mark.parametrize("op", ["cancel", "reject", "expire"])
def test_late_fill_is_a_stutter_on_every_terminal_status(op: str) -> None:
    book = step(_submitted(), {"op": op, "order_id": "A"})
    after = step(book, {"op": "late_fill", "order_id": "A", "fill_id": "f9"})
    assert after.orders["A"].status == book.orders["A"].status
    assert after.orders["A"].filled == 0.0


def test_unknown_action_is_refused() -> None:
    with pytest.raises(LifecycleViolation, match="unknown op explode"):
        step(Book(), {"op": "explode", "order_id": "A"})


# -- checkpoint / crash --


def test_checkpoint_then_crash_restores_the_snapshot() -> None:
    working = _submitted(4.0)
    checkpointed = step(working, {"op": "checkpoint"})
    mutated = step(
        checkpointed,
        {"op": "fill", "order_id": "A", "fill_id": "f1", "qty": 2.0, "rest": True},
    )
    assert mutated.orders["A"].filled == 2.0
    restarted = step(mutated, {"op": "crash"})
    assert restarted.orders["A"].filled == 0.0
    assert restarted.orders["A"].fills == {}
    assert restarted.orders["A"].status == "new"


def test_crash_without_checkpoint_drops_uncommitted_fills() -> None:
    # The spec admits the lossy crash: no snapshot means restore to absent.
    book = _filled()
    restarted = step(book, {"op": "crash"})
    assert restarted.orders["A"].status == "absent"
    assert restarted.orders["A"].filled == 0.0


# -- invariants --


def test_overfilled_order_is_reported() -> None:
    order = OrderState(status="filled", qty=1.0, filled=2.0, fills={"f1": 2.0})
    found = invariant_violations(order)
    assert "overfilled" in found
    assert "volatile filled order filled 2.0 of 1.0" in found


def test_fill_sum_mismatch_is_reported() -> None:
    order = OrderState(status="partial", qty=4.0, filled=1.0, fills={})
    assert "filled is not the sum of applied fills" in invariant_violations(order)


def test_durable_state_ahead_of_volatile_state_is_reported() -> None:
    order = OrderState(
        qty=4.0, filled=1.0, fills={"f1": 1.0}, s_qty=4.0, s_filled=2.0, s_fills={"f1": 2.0}
    )
    found = invariant_violations(order)
    assert "stable filled exceeds volatile filled" in found
    assert "stable fill f1 missing from volatile state" in found


def test_stable_overfill_is_reported() -> None:
    order = OrderState(
        status="new", qty=1.0, s_status="new", s_qty=1.0, s_filled=2.0, s_fills={"f1": 2.0}
    )
    found = invariant_violations(order)
    assert "stable overfilled" in found
    assert "stable filled is not the sum of stable fills" not in found


@pytest.mark.parametrize(
    ("status", "qty", "filled"),
    [
        ("absent", 1.0, 0.0),
        ("absent", 0.0, 1.0),
        ("new", 0.0, 0.0),
        ("new", 1.0, 1.0),
        ("partial", 1.0, 0.0),
        ("partial", 1.0, 1.0),
        ("filled", 1.0, 0.0),
        ("canceled", 1.0, 2.0),
        ("rejected", 1.0, 2.0),
        ("expired", 1.0, 2.0),
        ("bogus", 1.0, 0.0),
    ],
)
def test_shape_violations_are_reported(status: str, qty: float, filled: float) -> None:
    order = OrderState(
        status=status, qty=qty, filled=filled, fills={"f1": filled} if filled else {}
    )
    found = invariant_violations(order)
    assert found, (status, qty, filled)


def test_consistent_terminal_order_has_no_violations() -> None:
    order = OrderState(status="canceled", qty=1.0, s_status="canceled", s_qty=1.0)
    assert invariant_violations(order) == []


# -- traces and enumeration --


def test_check_trace_stops_at_the_first_violation() -> None:
    result = check_trace(
        [
            {"op": "submit", "order_id": "A", "qty": 2.0},
            {"op": "submit", "order_id": "A", "qty": 2.0},
            {"op": "cancel", "order_id": "A"},
        ]
    )
    assert result.ok is False
    assert len(result.violations) == 1
    assert result.violations[0].startswith("event 1 submit:")
    assert result.book.orders["A"].status == "new"


def test_check_trace_of_an_empty_list_is_ok() -> None:
    result = check_trace([])
    assert result.ok is True
    assert result.violations == []
    assert result.book.orders == {}


@pytest.mark.parametrize(("max_qty", "max_fill_id"), [(0, 1), (1, 0), (-1, -1)])
def test_enumeration_requires_positive_bounds(max_qty: int, max_fill_id: int) -> None:
    with pytest.raises(ValueError, match="bounds must be positive"):
        enumerate_safety(max_qty=max_qty, max_fill_id=max_fill_id)


def test_enumeration_reaches_more_than_the_initial_state() -> None:
    states, violations = enumerate_safety(max_qty=1, max_fill_id=1)
    assert violations == []
    assert states > 1
