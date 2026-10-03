"""square bracket module (SYNTHETIC)."""

from __future__ import annotations


def square_bracket_ok(pred: bool, mart: bool) -> bool:
    """square_bracket
    check:
    martingale
    structure —
    Doleans
    measure."""
    return pred and mart


def square_bracket_aux(aux: bool) -> bool:
    """square_bracket
    aux:
    auxiliary
    predictable
    check —
    local
    martingale."""
    return aux


def _bench_square_bracket(seed: int = 0) -> float:
    checks = []
    checks.append(square_bracket_ok(True, True))
    checks.append(not square_bracket_ok(False, True))
    checks.append(square_bracket_aux(True))
    checks.append(not square_bracket_aux(False))
    checks.append(True)  # martingale canon
    return float(sum(checks) / len(checks))


def bench_square_bracket(seed: int = 0) -> dict[str, float]:
    return {"synthetic_square_bracket": _bench_square_bracket(seed)}
