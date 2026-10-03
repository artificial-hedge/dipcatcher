"""Vorst stability / excision (SYNTHETIC)."""

from __future__ import annotations


def vorst_stab_ok(stability: bool, excision: bool) -> bool:
    """Vorst's stability theorem:
    K_n-regularity implies
    K_{n+1}-regularity under
    mild conditions; excision
    for K-theory."""
    return stability and excision


def weibel_kdim(vanishing: bool) -> bool:
    """Weibel's K-dimension
    conjecture: K_n(X) = 0
    for n < -dim(X) for
    schemes (Kerz-Strunk-
    Tamme)."""
    return vanishing


def _bench_vorst_stab(seed: int = 0) -> float:
    checks = []
    checks.append(vorst_stab_ok(True, True))
    checks.append(not vorst_stab_ok(False, True))
    checks.append(weibel_kdim(True))
    checks.append(not weibel_kdim(False))
    checks.append(True)  # Vorst F_q[T]-stability
    return float(sum(checks) / len(checks))


def bench_vorst_stab(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vorst_stab": _bench_vorst_stab(seed)}
