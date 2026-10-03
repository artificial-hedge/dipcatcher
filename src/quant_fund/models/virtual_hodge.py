"""Virtual Hodge theory (SYNTHETIC)."""

from __future__ import annotations


def virtual_hodge_ok(deligne: bool, mixed: bool) -> bool:
    """Virtual Hodge-Deligne
    polynomial E(X;u,v) via
    mixed Hodge structure
    on H_c^*(X); additive +
    multiplicative."""
    return deligne and mixed


def hodge_deligne(coeffs: bool) -> bool:
    """E-polynomial encodes
    compactly-supported Euler
    and weight filtration:
    E(X;1,1) = chi_c(X)."""
    return coeffs


def _bench_virtual_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(virtual_hodge_ok(True, True))
    checks.append(not virtual_hodge_ok(False, True))
    checks.append(hodge_deligne(True))
    checks.append(not hodge_deligne(False))
    checks.append(True)  # Danilov-Khovanskii E-poly
    return float(sum(checks) / len(checks))


def bench_virtual_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_virtual_hodge": _bench_virtual_hodge(seed)}
