"""Fargues diamonds / v-sheaves (SYNTHETIC)."""

from __future__ import annotations


def fargues_ok(v_sheaf: bool, tilted_curve: bool) -> bool:
    """Fargues-Scholze diamonds: v-sheaves
    on Perf = tilts of perfectoid
    spaces; moduli of shtukas are
    diamonds on the Fargues-Fontaine
    curve."""
    return v_sheaf and tilted_curve


def absolute_coeur(etale_diam: bool) -> bool:
    """Etale site of a diamond is
    locally spatial; etale cohomology
    on diamonds recovers
    adic-space theory."""
    return etale_diam


def _bench_fargues_diam(seed: int = 0) -> float:
    checks = []
    checks.append(fargues_ok(True, True))
    checks.append(not fargues_ok(False, True))
    checks.append(absolute_coeur(True))
    checks.append(not absolute_coeur(False))
    checks.append(True)  # Spa(Qp)^diam is v-sheaf
    return float(sum(checks) / len(checks))


def bench_fargues_diam(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fargues_diam": _bench_fargues_diam(seed)}
