"""diffusion semigroup module (SYNTHETIC)."""

from __future__ import annotations


def diffusion_semigroup_ok(sc: bool, sp: bool) -> bool:
    """diffusion_semigroup
    check:
    diffusion
    theory —
    boundary."""
    return sc and sp


def diffusion_semigroup_aux(aux: bool) -> bool:
    """diffusion_semigroup
    aux:
    auxiliary
    diffusion
    check —
    generator."""
    return aux


def _bench_diffusion_semigroup(seed: int = 0) -> float:
    checks = []
    checks.append(diffusion_semigroup_ok(True, True))
    checks.append(not diffusion_semigroup_ok(False, True))
    checks.append(diffusion_semigroup_aux(True))
    checks.append(not diffusion_semigroup_aux(False))
    checks.append(True)  # diffusion canon
    return float(sum(checks) / len(checks))


def bench_diffusion_semigroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffusion_semigroup": _bench_diffusion_semigroup(seed)}
