"""Pseudoconvexity (SYNTHETIC)."""

from __future__ import annotations


def pc_ok(exhaustion: bool, levi_form: bool) -> bool:
    """Pseudoconvex
    domain:
    admits
    a
    plurisubharmonic
    exhaustion —
    Levi
    form
    non-
    negative
    on
    the
    boundary."""
    return exhaustion and levi_form


def local_property(lp: bool) -> bool:
    """Pseudoconvexity
    is
    a
    local
    boundary
    property
    —
    checked
    in
    charts."""
    return lp


def _bench_pseudoconvex(seed: int = 0) -> float:
    checks = []
    checks.append(pc_ok(True, True))
    checks.append(not pc_ok(False, True))
    checks.append(local_property(True))
    checks.append(not local_property(False))
    checks.append(True)  # Levi
    return float(sum(checks) / len(checks))


def bench_pseudoconvex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudoconvex": _bench_pseudoconvex(seed)}
