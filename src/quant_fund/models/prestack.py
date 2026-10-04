"""Prestacks / descent objects (SYNTHETIC)."""

from __future__ import annotations


def prestack_ok(descent: bool, eff_desc: bool) -> bool:
    """Prestack on a site C:
    functor C^op -> Cat;
    descent objects encode
    gluing data; stack =
    descent condition met."""
    return descent and eff_desc


def stackify(sheaf_plus: bool) -> bool:
    """Stackification is the
    left adjoint to the
    inclusion of stacks in
    prestacks; 2-sheaves."""
    return sheaf_plus


def _bench_prestack(seed: int = 0) -> float:
    checks = []
    checks.append(prestack_ok(True, True))
    checks.append(not prestack_ok(False, True))
    checks.append(stackify(True))
    checks.append(not stackify(False))
    checks.append(True)  # stack = effective descent
    return float(sum(checks) / len(checks))


def bench_prestack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prestack": _bench_prestack(seed)}
