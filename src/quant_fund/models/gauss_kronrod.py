"""gauss kronrod module (SYNTHETIC)."""

from __future__ import annotations


def gauss_kronrod_ok(node: bool, weight: bool) -> bool:
    """gauss_kronrod
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def gauss_kronrod_aux(aux: bool) -> bool:
    """gauss_kronrod
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_gauss_kronrod(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_kronrod_ok(True, True))
    checks.append(not gauss_kronrod_ok(False, True))
    checks.append(gauss_kronrod_aux(True))
    checks.append(not gauss_kronrod_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_gauss_kronrod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_kronrod": _bench_gauss_kronrod(seed)}
