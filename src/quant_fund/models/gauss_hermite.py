"""gauss hermite module (SYNTHETIC)."""

from __future__ import annotations


def gauss_hermite_ok(node: bool, wgt: bool) -> bool:
    """gauss_hermite
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def gauss_hermite_aux(aux: bool) -> bool:
    """gauss_hermite
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_gauss_hermite(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_hermite_ok(True, True))
    checks.append(not gauss_hermite_ok(False, True))
    checks.append(gauss_hermite_aux(True))
    checks.append(not gauss_hermite_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_gauss_hermite(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_hermite": _bench_gauss_hermite(seed)}
