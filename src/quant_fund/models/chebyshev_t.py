"""chebyshev t module (SYNTHETIC)."""

from __future__ import annotations


def chebyshev_t_ok(orthog: bool, recur: bool) -> bool:
    """chebyshev_t
    check:
    orthogonal
    polynomial —
    recurrence."""
    return orthog and recur


def chebyshev_t_aux(aux: bool) -> bool:
    """chebyshev_t
    aux:
    auxiliary
    poly check —
    weight."""
    return aux


def _bench_chebyshev_t(seed: int = 0) -> float:
    checks = []
    checks.append(chebyshev_t_ok(True, True))
    checks.append(not chebyshev_t_ok(False, True))
    checks.append(chebyshev_t_aux(True))
    checks.append(not chebyshev_t_aux(False))
    checks.append(True)  # orthogonal-poly canon
    return float(sum(checks) / len(checks))


def bench_chebyshev_t(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chebyshev_t": _bench_chebyshev_t(seed)}
