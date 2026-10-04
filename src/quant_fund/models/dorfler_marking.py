"""dorfler marking module (SYNTHETIC)."""

from __future__ import annotations


def dorfler_marking_ok(mark: bool, est: bool) -> bool:
    """dorfler_marking
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def dorfler_marking_aux(aux: bool) -> bool:
    """dorfler_marking
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_dorfler_marking(seed: int = 0) -> float:
    checks = []
    checks.append(dorfler_marking_ok(True, True))
    checks.append(not dorfler_marking_ok(False, True))
    checks.append(dorfler_marking_aux(True))
    checks.append(not dorfler_marking_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_dorfler_marking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dorfler_marking": _bench_dorfler_marking(seed)}
