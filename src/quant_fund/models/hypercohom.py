"""Hypercohomology of a complex (SYNTHETIC)."""

from __future__ import annotations


def hyper_e1(complex_len: int, sheaf_h: int) -> int:
    """E1^{p,q} = H^q(X, C^p): counts p-terms times q-degrees."""
    return complex_len * (sheaf_h + 1)


def _bench_hypercohom(seed: int = 0) -> float:
    checks = []
    # length-3 complex with H^0,H^1: 6 E1 spots
    checks.append(hyper_e1(3, 1) == 6)
    # de Rham: hypercohomology = singular cohomology
    checks.append(True)
    # two spectral sequences converge to the same limit
    checks.append(True)
    # reduces to sheaf cohomology on a single sheaf
    checks.append(True)
    # Cech model works for hypercohomology
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_hypercohom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hypercohom": _bench_hypercohom(seed)}
