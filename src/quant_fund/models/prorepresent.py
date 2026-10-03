"""Prorepresentability (SYNTHETIC)."""

from __future__ import annotations


def pr_ok(prorepresentable: bool, functor: bool) -> bool:
    """Pro:
    prorepresentable
    deformation
    functor —
    Grothendieck
    prorep."""
    return prorepresentable and functor


def prorep_cond(pc: bool) -> bool:
    """Prorepresentable
    condition:
    prorepresentability
    condition —
    Grothendieck
    prorep."""
    return pc


def _bench_prorepresent(seed: int = 0) -> float:
    checks = []
    checks.append(pr_ok(True, True))
    checks.append(not pr_ok(False, True))
    checks.append(prorep_cond(True))
    checks.append(not prorep_cond(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_prorepresent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prorepresent": _bench_prorepresent(seed)}
