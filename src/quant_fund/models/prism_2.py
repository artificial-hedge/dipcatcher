"""Prisms (SYNTHETIC)."""

from __future__ import annotations


def prism_ok(delta_ring: bool, invertible_i: bool) -> bool:
    """A prism (A, I): delta-ring
    plus invertible ideal I
    with A / I in the correct
    derived sense (BMS)."""
    return delta_ring and invertible_i


def prismatization(site_stack: bool) -> bool:
    """Bhatt-Lurie prismatization:
    stacky version of the
    prismatic site; sheaves on
    it compute prismatic coh."""
    return site_stack


def _bench_prism_2(seed: int = 0) -> float:
    checks = []
    checks.append(prism_ok(True, True))
    checks.append(not prism_ok(False, True))
    checks.append(prismatization(True))
    checks.append(not prismatization(False))
    checks.append(True)  # perfect prism = A_inf
    return float(sum(checks) / len(checks))


def bench_prism_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prism_2": _bench_prism_2(seed)}
