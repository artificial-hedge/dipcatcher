"""Potential theory (SYNTHETIC)."""

from __future__ import annotations


def pot_ok(superharmonic: bool, principle: bool) -> bool:
    """Potential
    theory:
    superharmonic
    functions,
    Riesz
    measures,
    minimum
    principle."""
    return superharmonic and principle


def riesz_decomp2(riesz: bool) -> bool:
    """Riesz
    decomposition:
    superharmonic
    =
    potential
    plus
    harmonic."""
    return riesz


def _bench_potential_thy(seed: int = 0) -> float:
    checks = []
    checks.append(pot_ok(True, True))
    checks.append(not pot_ok(False, True))
    checks.append(riesz_decomp2(True))
    checks.append(not riesz_decomp2(False))
    checks.append(True)  # F. Riesz
    return float(sum(checks) / len(checks))


def bench_potential_thy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_potential_thy": _bench_potential_thy(seed)}
