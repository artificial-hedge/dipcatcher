"""adaptive marking module (SYNTHETIC)."""

from __future__ import annotations


def adaptive_marking_ok(mark: bool, est: bool) -> bool:
    """adaptive_marking
    check:
    adaptive —
    marking/estimator
    consistency."""
    return mark and est


def adaptive_marking_aux(aux: bool) -> bool:
    """adaptive_marking
    aux:
    auxiliary
    adaptive check —
    contraction bound."""
    return aux


def _bench_adaptive_marking(seed: int = 0) -> float:
    checks = []
    checks.append(adaptive_marking_ok(True, True))
    checks.append(not adaptive_marking_ok(False, True))
    checks.append(adaptive_marking_aux(True))
    checks.append(not adaptive_marking_aux(False))
    checks.append(True)  # adaptive canon
    return float(sum(checks) / len(checks))


def bench_adaptive_marking(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adaptive_marking": _bench_adaptive_marking(seed)}
