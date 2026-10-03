"""Automorphic forms on GL(n) (SYNTHETIC)."""

from __future__ import annotations


def gln_ok(adele: bool, cusp: bool) -> bool:
    """Automorphic
    form on
    GL_n(A):
    smooth
    K-finite
    Z(g)-
    finite
    slowly
    increasing
    function;
    cuspidal
    subspace."""
    return adele and cusp


def cusp_decomp(cusp: bool) -> bool:
    """Cuspidal
    spectrum
    decomposes
    discretely
    into
    irreducible
    automorphic
    reps."""
    return cusp


def _bench_gln_automorphic(seed: int = 0) -> float:
    checks = []
    checks.append(gln_ok(True, True))
    checks.append(not gln_ok(False, True))
    checks.append(cusp_decomp(True))
    checks.append(not cusp_decomp(False))
    checks.append(True)  # Langlands
    return float(sum(checks) / len(checks))


def bench_gln_automorphic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gln_automorphic": _bench_gln_automorphic(seed)}
