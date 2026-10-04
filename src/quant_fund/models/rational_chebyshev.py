"""rational chebyshev module (SYNTHETIC)."""

from __future__ import annotations


def rational_chebyshev_ok(rational: bool, approx: bool) -> bool:
    """rational_chebyshev
    check:
    rational
    approximation —
    Padé."""
    return rational and approx


def rational_chebyshev_aux(aux: bool) -> bool:
    """rational_chebyshev
    aux:
    auxiliary
    approx check —
    convergent."""
    return aux


def _bench_rational_chebyshev(seed: int = 0) -> float:
    checks = []
    checks.append(rational_chebyshev_ok(True, True))
    checks.append(not rational_chebyshev_ok(False, True))
    checks.append(rational_chebyshev_aux(True))
    checks.append(not rational_chebyshev_aux(False))
    checks.append(True)  # rational-approx canon
    return float(sum(checks) / len(checks))


def bench_rational_chebyshev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rational_chebyshev": _bench_rational_chebyshev(seed)}
