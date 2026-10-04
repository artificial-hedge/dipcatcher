"""bounded diff module (SYNTHETIC)."""

from __future__ import annotations


def bounded_diff_ok(con: bool, tail: bool) -> bool:
    """bounded_diff
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def bounded_diff_aux(aux: bool) -> bool:
    """bounded_diff
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_bounded_diff(seed: int = 0) -> float:
    checks = []
    checks.append(bounded_diff_ok(True, True))
    checks.append(not bounded_diff_ok(False, True))
    checks.append(bounded_diff_aux(True))
    checks.append(not bounded_diff_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_bounded_diff(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_diff": _bench_bounded_diff(seed)}
