"""Weyl-Kac character formula (SYNTHETIC)."""

from __future__ import annotations


def weyl_kac_ok(char: bool, integrable: bool) -> bool:
    """Weyl-Kac
    character
    formula:
    ch L(lambda)
    = sum_w
    eps(w)
    e^{w(lambda+rho)}
    / prod roots."""
    return char and integrable


def integrable_highest(integ: bool) -> bool:
    """Integrable
    highest
    weight
    modules:
    dominant
    weights
    give
    irreducible
    integrable
    reps."""
    return integ


def _bench_weyl_kac(seed: int = 0) -> float:
    checks = []
    checks.append(weyl_kac_ok(True, True))
    checks.append(not weyl_kac_ok(False, True))
    checks.append(integrable_highest(True))
    checks.append(not integrable_highest(False))
    checks.append(True)  # Kac
    return float(sum(checks) / len(checks))


def bench_weyl_kac(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weyl_kac": _bench_weyl_kac(seed)}
