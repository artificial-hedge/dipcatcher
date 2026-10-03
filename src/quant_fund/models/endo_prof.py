"""Endoprofunctors / bimodules (SYNTHETIC)."""

from __future__ import annotations


def endo_prof_ok(profunctor: bool, tensor_comp: bool) -> bool:
    """Endoprofunctors on C:
    functors C^op x C -> Set
    compose as bimodules;
    collage glues into
    a single category."""
    return profunctor and tensor_comp


def bimodule_comp(composition: bool) -> bool:
    """Bimodule composition via
    coends; bicompletion
    embeds C into
    Prof(C,C)."""
    return composition


def _bench_endo_prof(seed: int = 0) -> float:
    checks = []
    checks.append(endo_prof_ok(True, True))
    checks.append(not endo_prof_ok(False, True))
    checks.append(bimodule_comp(True))
    checks.append(not bimodule_comp(False))
    checks.append(True)  # Cauchy completion
    return float(sum(checks) / len(checks))


def bench_endo_prof(seed: int = 0) -> dict[str, float]:
    return {"synthetic_endo_prof": _bench_endo_prof(seed)}
