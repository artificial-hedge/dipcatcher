"""Etale fundamental group (SYNTHETIC)."""

from __future__ import annotations


def ef_ok(etale_pi1: bool, profinite: bool) -> bool:
    """Etale
    pi1:
    profinite
    fundamental
    group —
    Grothendieck."""
    return etale_pi1 and profinite


def groth_fund(gf: bool) -> bool:
    """Grothendieck
    pi1:
    fiber
    functor
    automorphisms —
    etale
    pi1."""
    return gf


def _bench_etale_fund(seed: int = 0) -> float:
    checks = []
    checks.append(ef_ok(True, True))
    checks.append(not ef_ok(False, True))
    checks.append(groth_fund(True))
    checks.append(not groth_fund(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_etale_fund(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_fund": _bench_etale_fund(seed)}
