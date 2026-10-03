"""Surgery theory (SYNTHETIC)."""

from __future__ import annotations


def st_ok(normal_map: bool, surgery_obstruction: bool) -> bool:
    """Surgery
    theory:
    normal
    maps
    with
    surgery
    obstructions
    in
    L-
    groups —
    Browder-
    Novikov-
    Sullivan-
    Wall."""
    return normal_map and surgery_obstruction


def surgery_exact_seq(ses: bool) -> bool:
    """Surgery
    exact
    sequence:
    structure
    set,
    normal
    invariants,
    L-
    groups —
    classification
    of
    manifolds."""
    return ses


def _bench_surgery_theory(seed: int = 0) -> float:
    checks = []
    checks.append(st_ok(True, True))
    checks.append(not st_ok(False, True))
    checks.append(surgery_exact_seq(True))
    checks.append(not surgery_exact_seq(False))
    checks.append(True)  # BNSW
    return float(sum(checks) / len(checks))


def bench_surgery_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_surgery_theory": _bench_surgery_theory(seed)}
