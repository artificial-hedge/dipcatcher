"""Dyson Brownian motion (SYNTHETIC)."""

from __future__ import annotations


def db_ok(repulsion: bool, eigenvalue_flow: bool) -> bool:
    """Dyson
    Brownian:
    eigenvalue
    SDE
    with
    Vandermonde
    repulsion —
    dynamic
    GUE."""
    return repulsion and eigenvalue_flow


def local_relaxation(lr: bool) -> bool:
    """Local
    relaxation:
    DBM
    reaches
    local
    equilibrium
    in
    short
    time —
    universality
    tool."""
    return lr


def _bench_dyson_brownian(seed: int = 0) -> float:
    checks = []
    checks.append(db_ok(True, True))
    checks.append(not db_ok(False, True))
    checks.append(local_relaxation(True))
    checks.append(not local_relaxation(False))
    checks.append(True)  # Dyson-Erdos-Yau
    return float(sum(checks) / len(checks))


def bench_dyson_brownian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dyson_brownian": _bench_dyson_brownian(seed)}
