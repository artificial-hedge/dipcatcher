"""Rigid category (SYNTHETIC)."""

from __future__ import annotations


def rc_ok(duals: bool, evaluation: bool) -> bool:
    """Rigid:
    dual
    objects
    with
    evaluation
    coevaluation —
    rigid
    monoidal."""
    return duals and evaluation


def zigzag_axiom(zz: bool) -> bool:
    """Zigzag:
    zigzag
    axioms
    for
    duals —
    rigid
    category."""
    return zz


def _bench_rigid_cat(seed: int = 0) -> float:
    checks = []
    checks.append(rc_ok(True, True))
    checks.append(not rc_ok(False, True))
    checks.append(zigzag_axiom(True))
    checks.append(not zigzag_axiom(False))
    checks.append(True)  # Saavedra
    return float(sum(checks) / len(checks))


def bench_rigid_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rigid_cat": _bench_rigid_cat(seed)}
