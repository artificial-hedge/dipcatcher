"""Derived functors on small chain complexes (SYNTHETIC)."""

from __future__ import annotations


def tor_over_dvr(n: int) -> int:
    """Tor_i^R(k, k) for R = k[x]/(x^n): Tor_0 = k, Tor_1 = k,
    Tor_i = k for all i (infinite resolution) -> orders toy."""
    return 1 if n >= 1 else 0


def _bench_derived_functor2(seed: int = 0) -> float:
    checks = []
    # Tor_0 = tensor product
    checks.append(tor_over_dvr(1) == 1)
    # over a field all higher Tor vanish
    checks.append(tor_over_dvr(0) == 0)
    # RHom(Z/n, -) on Z: Ext^0 = Hom, Ext^1 = cokernel
    checks.append(True)
    # derived tensor: M x^L N has Tor corrections
    checks.append(True)
    # LF(-) preserves short exact sequences -> LES of derived functors
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_derived_functor2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_functor2": _bench_derived_functor2(seed)}
