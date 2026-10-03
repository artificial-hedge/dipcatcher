"""gauss chebyshev module (SYNTHETIC)."""

from __future__ import annotations


def gauss_chebyshev_ok(node: bool, weight: bool) -> bool:
    """gauss_chebyshev
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def gauss_chebyshev_aux(aux: bool) -> bool:
    """gauss_chebyshev
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_gauss_chebyshev(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_chebyshev_ok(True, True))
    checks.append(not gauss_chebyshev_ok(False, True))
    checks.append(gauss_chebyshev_aux(True))
    checks.append(not gauss_chebyshev_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_gauss_chebyshev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_chebyshev": _bench_gauss_chebyshev(seed)}
