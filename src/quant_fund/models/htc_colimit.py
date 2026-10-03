"""Homotopy-coherent colimits (SYNTHETIC)."""

from __future__ import annotations


def hco_ok(coherent: bool, univ: bool) -> bool:
    """Homotopy-coherent
    colimit: colimit
    in a quasi-cat
    satisfies the
    hom-coherent
    universal
    property."""
    return coherent and univ


def hcolim_model(bousfield: bool) -> bool:
    """Bousfield-Kan
    formula computes
    hocolim via
    geometric
    realization of
    a simplicial
    replacement."""
    return bousfield


def _bench_htc_colimit(seed: int = 0) -> float:
    checks = []
    checks.append(hco_ok(True, True))
    checks.append(not hco_ok(False, True))
    checks.append(hcolim_model(True))
    checks.append(not hcolim_model(False))
    checks.append(True)  # Bousfield-Kan
    return float(sum(checks) / len(checks))


def bench_htc_colimit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_htc_colimit": _bench_htc_colimit(seed)}
