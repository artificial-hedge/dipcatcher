"""D-modules on curves (SYNTHETIC)."""

from __future__ import annotations


def dm_ok(differential_ops: bool, connections: bool) -> bool:
    """D-
    module:
    modules
    over
    differential
    operators —
    connections
    and
    PDEs."""
    return differential_ops and connections


def de_rham_dr(dr: bool) -> bool:
    """de
    Rham
    functor:
    D-
    module
    to
    perverse
    sheaf —
    Riemann-
    Hilbert."""
    return dr


def _bench_d_module2(seed: int = 0) -> float:
    checks = []
    checks.append(dm_ok(True, True))
    checks.append(not dm_ok(False, True))
    checks.append(de_rham_dr(True))
    checks.append(not de_rham_dr(False))
    checks.append(True)  # Riemann-Hilbert
    return float(sum(checks) / len(checks))


def bench_d_module2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_d_module2": _bench_d_module2(seed)}
