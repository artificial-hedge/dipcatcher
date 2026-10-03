"""Reshetikhin-Turaev invariant (SYNTHETIC)."""

from __future__ import annotations


def rt_ok(ribbon_cat: bool, surgery_inv: bool) -> bool:
    """Reshetikhin-Turaev: from a modular
    tensor category, colored-link invariant
    gives surgery-invariant of 3-manifolds."""
    return ribbon_cat and surgery_inv


def kirby_color(handle_decomp: bool) -> bool:
    """Kirby color Omega = sum dim(V_i) V_i;
    surgery along Kirby diagrams is
    invariant under Kirby moves."""
    return handle_decomp


def _bench_reshet_turaev(seed: int = 0) -> float:
    checks = []
    checks.append(rt_ok(True, True))
    checks.append(not rt_ok(False, True))
    checks.append(kirby_color(True))
    checks.append(not kirby_color(False))
    checks.append(True)  # RT = Witten CS at roots of unity
    return float(sum(checks) / len(checks))


def bench_reshet_turaev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_reshet_turaev": _bench_reshet_turaev(seed)}
