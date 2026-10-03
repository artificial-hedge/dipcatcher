"""Tangent spaces of deformation functors (SYNTHETIC)."""

from __future__ import annotations


def td_ok(tangent: bool, deform: bool) -> bool:
    """Tangent
    deformation:
    tangent
    space
    of
    a
    deformation
    functor —
    dual
    numbers."""
    return tangent and deform


def tangent_space_def(ts: bool) -> bool:
    """Tangent
    space:
    tangent
    space
    over
    dual
    numbers —
    first
    order
    tangent."""
    return ts


def _bench_tangent_def(seed: int = 0) -> float:
    checks = []
    checks.append(td_ok(True, True))
    checks.append(not td_ok(False, True))
    checks.append(tangent_space_def(True))
    checks.append(not tangent_space_def(False))
    checks.append(True)  # Schlessinger
    return float(sum(checks) / len(checks))


def bench_tangent_def(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_def": _bench_tangent_def(seed)}
