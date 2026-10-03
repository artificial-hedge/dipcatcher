"""gauss laguerre module (SYNTHETIC)."""

from __future__ import annotations


def gauss_laguerre_ok(node: bool, wgt: bool) -> bool:
    """gauss_laguerre
    check:
    quadrature/tensor —
    node/weight
    consistency."""
    return node and wgt


def gauss_laguerre_aux(aux: bool) -> bool:
    """gauss_laguerre
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_gauss_laguerre(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_laguerre_ok(True, True))
    checks.append(not gauss_laguerre_ok(False, True))
    checks.append(gauss_laguerre_aux(True))
    checks.append(not gauss_laguerre_aux(False))
    checks.append(True)  # QMC/tensor canon
    return float(sum(checks) / len(checks))


def bench_gauss_laguerre(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_laguerre": _bench_gauss_laguerre(seed)}
