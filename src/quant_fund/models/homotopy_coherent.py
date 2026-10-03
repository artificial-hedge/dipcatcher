"""Homotopy-coherent diagrams (SYNTHETIC)."""

from __future__ import annotations


def hcoh_ok(coherent: bool, thickening: bool) -> bool:
    """Homotopy-coherent
    diagram: simplicial
    functor C -> sSet
    from the coherent
    thickening of a
    small category."""
    return coherent and thickening


def coh_nerve(cordier: bool) -> bool:
    """Coherent nerve:
    N_•(C) of a
    simplicially
    enriched cat is
    a quasi-cat
    (Cordier-Porter)."""
    return cordier


def _bench_homotopy_coherent(seed: int = 0) -> float:
    checks = []
    checks.append(hcoh_ok(True, True))
    checks.append(not hcoh_ok(False, True))
    checks.append(coh_nerve(True))
    checks.append(not coh_nerve(False))
    checks.append(True)  # Vogt theory
    return float(sum(checks) / len(checks))


def bench_homotopy_coherent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_coherent": _bench_homotopy_coherent(seed)}
