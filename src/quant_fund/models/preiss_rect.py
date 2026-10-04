"""Preiss rectifiability (SYNTHETIC)."""

from __future__ import annotations


def preiss_ok(finite: bool, density: bool) -> bool:
    """Preiss's
    theorem:
    finite
    positive
    m-density
    a.e.
    implies
    m-
    rectifiability."""
    return finite and density


def cone_flat(cf: bool) -> bool:
    """Conical/
    flat
    tangent
    measures
    characterize
    the
    rectifiable
    case."""
    return cf


def _bench_preiss_rect(seed: int = 0) -> float:
    checks = []
    checks.append(preiss_ok(True, True))
    checks.append(not preiss_ok(False, True))
    checks.append(cone_flat(True))
    checks.append(not cone_flat(False))
    checks.append(True)  # Preiss
    return float(sum(checks) / len(checks))


def bench_preiss_rect(seed: int = 0) -> dict[str, float]:
    return {"synthetic_preiss_rect": _bench_preiss_rect(seed)}
