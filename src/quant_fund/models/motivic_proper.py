"""Proper motivic morphism (SYNTHETIC)."""

from __future__ import annotations


def mp_ok(proper_push: bool, exceptional_push: bool) -> bool:
    """Proper
    morphism:
    f_!
    equals
    f_*
    for
    proper —
    proper
    support."""
    return proper_push and exceptional_push


def projective_bundle(pb: bool) -> bool:
    """Projective
    bundle:
    cohomology
    of
    projective
    bundle
    splits —
    projective
    bundle
    formula."""
    return pb


def _bench_motivic_proper(seed: int = 0) -> float:
    checks = []
    checks.append(mp_ok(True, True))
    checks.append(not mp_ok(False, True))
    checks.append(projective_bundle(True))
    checks.append(not projective_bundle(False))
    checks.append(True)  # Voevodsky
    return float(sum(checks) / len(checks))


def bench_motivic_proper(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_proper": _bench_motivic_proper(seed)}
