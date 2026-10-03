"""Gradual typing: consistent-equality (~), casts, and blame tracking.

Types: ("int",) ("bool",) ("fun",A,B) ("dyn",). Consistent-equality treats
dyn as matching everything both ways. Cast insertion: e::A⇒B wraps the
value with source blame label; a failing cast blames the side that
asserted the impossible precision (positive/negative blame in the
Wadler–Findler sense).
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1028

Ty = tuple
DYN: Ty = ("dyn",)


def ceq(a: Ty, b: Ty) -> bool:
    """Consistent equality ~ : dyn ~ T both ways; fun components pointwise."""
    if a == DYN or b == DYN:
        return True
    if a[0] != b[0]:
        return False
    if a[0] == "fun":
        return ceq(a[1], b[1]) and ceq(a[2], b[2])
    return a == b


def precision(a: Ty, b: Ty) -> bool:
    """a ⊑ b: a is less precise (more dyn) than b."""
    if a == DYN:
        return True
    if b == DYN:
        return False
    if a == b:
        return True
    if a[0] == "fun" and b[0] == "fun":
        return precision(b[1], a[1]) and precision(a[2], b[2])
    return False


class Blame(Exception):
    def __init__(self, label: str) -> None:
        self.label = label


def cast(v: Any, a: Ty, b: Ty, label: str = "+") -> Any:
    """Apply cast a⇒b with blame `label` on failure."""
    if a == b or b == DYN:
        return v
    if a == DYN:
        # runtime check that v actually has type b
        if _has_type(v, b):
            return v
        raise Blame(label)
    if a[0] == "fun" and b[0] == "fun":
        # wrap higher-order cast
        def wrapped(x: Any) -> Any:
            return cast(v(cast(x, b[1], a[1], _flip(label)), a[2], b[2], label), a[2], b[2], label)

        return wrapped
    raise Blame(label)


def _flip(lbl: str) -> str:
    return "-" if lbl == "+" else "+"


def _has_type(v: Any, t: Ty) -> bool:
    if t == DYN:
        return True
    if t == ("int",):
        return isinstance(v, int) and not isinstance(v, bool)
    if t == ("bool",):
        return isinstance(v, bool)
    if t[0] == "fun":
        return callable(v)
    return False


def bench_gradual_types(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    checks.append(ceq(DYN, ("int",)) and ceq(("int",), DYN))
    checks.append(not ceq(("int",), ("bool",)))
    checks.append(precision(DYN, ("int",)))
    checks.append(not precision(("int",), DYN))
    # successful dyn->int cast
    checks.append(cast(5, DYN, ("int",), "+") == 5)
    # failing cast blames the label
    try:
        cast(True, DYN, ("int",), "L1")
        ok = False
    except Blame as e:
        ok = e.label == "L1"
    checks.append(ok)
    # higher-order cast: dyn->(int->int) applied to int works
    f = cast(lambda x: x + 1, DYN, ("fun", ("int",), ("int",)), "+")
    checks.append(f(3) == 4)
    return {"synthetic_gradual_types": float(sum(checks)) / len(checks)}
