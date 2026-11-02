"""Apply the operator kill file to a paper book.

Live brokers are not passed here. ``HALT_NEW_ORDERS`` cancels resting paper
orders only when ``DIPCATCHER_HALT_CANCELS_OPEN_ORDERS`` is set. ``FLATTEN``
closes the simulated book. ``FLATTEN_OPTIONAL`` still requires a human and
does not move the book. A symlink, directory, or unknown token changes nothing.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_fund.utils.operator_kill import (
    CANCEL_OPEN_ORDERS,
    ENABLED,
    ENV_PATH,
    FLATTEN,
    FLATTEN_OPTIONAL,
    HALT_NEW_ORDERS,
    read_operator_kill,
)

_HALT_CANCEL_ENV = "DIPCATCHER_HALT_CANCELS_OPEN_ORDERS"
_TRUE = frozenset({"1", "true", "yes", "on"})


def halt_cancels_open_orders() -> bool:
    return os.environ.get(_HALT_CANCEL_ENV, "").strip().lower() in _TRUE


def _result(
    action: str,
    *,
    ok: bool = True,
    problem: str | None = None,
    cancelled: list[str] | None = None,
    names: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "ok": ok,
        "problem": problem,
        "cancelled": [] if cancelled is None else list(cancelled),
        "names": [] if names is None else list(names),
    }


def _cancel_action(book: object, action: str) -> dict[str, Any]:
    cancel = getattr(book, "cancel_operator_orders", None)
    if not callable(cancel):
        return _result("blocked", ok=False, problem="paper_book_unsupported")
    cancelled = cancel()
    return _result(action, cancelled=list(cancelled))


def _flatten_action(
    book: object, *, at: datetime | None, marks: dict[str, float] | None
) -> dict[str, Any]:
    flatten = getattr(book, "flatten_operator_book", None)
    if not callable(flatten):
        return _result("blocked", ok=False, problem="paper_book_unsupported")
    report = flatten(marks, at=at)
    problem = report.get("problem")
    return _result(
        "flatten",
        ok=bool(report.get("ok")),
        problem=problem if isinstance(problem, str) else None,
        cancelled=list(report.get("cancelled") or []),
        names=list(report.get("names") or []),
    )


def _dispatch(
    book: object, state: str, *, at: datetime | None, marks: dict[str, float] | None
) -> dict[str, Any]:
    if state == ENABLED:
        return _result("enabled")
    if state == FLATTEN_OPTIONAL:
        return _result("halt")
    if state == HALT_NEW_ORDERS and not halt_cancels_open_orders():
        return _result("halt")
    if state == HALT_NEW_ORDERS:
        return _cancel_action(book, "halt")
    if state == CANCEL_OPEN_ORDERS:
        return _cancel_action(book, "cancel")
    if state == FLATTEN:
        return _flatten_action(book, at=at, marks=marks)
    return _result("blocked", ok=False, problem="kill_switch_invalid")


def enforce_operator_book(
    book: object, *, at: datetime | None = None, marks: dict[str, float] | None = None
) -> dict[str, Any]:
    """Mutate ``book`` only for a recognized paper action. Otherwise leave it."""
    raw = os.environ.get(ENV_PATH, "").strip()
    if not raw:
        return _result("unconfigured")
    reading = read_operator_kill(Path(raw))
    if reading.problem is not None:
        return _result("blocked", ok=False, problem=reading.problem)
    return _dispatch(book, reading.state or ENABLED, at=at, marks=marks)
