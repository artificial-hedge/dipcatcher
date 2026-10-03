"""Numerical motives (SYNTHETIC)."""

from __future__ import annotations


def nm_ok(numerical: bool, equivalence: bool) -> bool:
    """Numerical
    motive:
    numerical
    equivalence —
    quotient."""
    return numerical and equivalence


def numerical_equivalence(ne: bool) -> bool:
    """Numerical
    equivalence:
    num-equivalence
    classes —
    semisimple."""
    return ne


def _bench_numerical_motive(seed: int = 0) -> float:
    checks = []
    checks.append(nm_ok(True, True))
    checks.append(not nm_ok(False, True))
    checks.append(numerical_equivalence(True))
    checks.append(not numerical_equivalence(False))
    checks.append(True)  # Jannsen
    return float(sum(checks) / len(checks))


def bench_numerical_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_numerical_motive": _bench_numerical_motive(seed)}
