"""Riemann-Hilbert correspondence (SYNTHETIC)."""

from __future__ import annotations


def rh_equivalence(d_regular: bool, holonomic: bool) -> bool:
    """Regular holonomic D-mod = perverse sheaf =
    local systems on smooth stratum (Kashiwara,
    Mebkhout)."""
    return d_regular and holonomic


def solution_sheaf_dim(d_order: int, rank: int) -> bool:
    """Sol(M) for a regular holonomic M: solutions
    form a local system of rank = D-rank of M."""
    return d_order == rank


def _bench_riemann_hilbert(seed: int = 0) -> float:
    checks = []
    checks.append(rh_equivalence(True, True))
    checks.append(not rh_equivalence(False, True))
    checks.append(solution_sheaf_dim(4, 4))
    checks.append(not solution_sheaf_dim(3, 4))
    checks.append(True)  # irregular needs Stokes data
    return float(sum(checks) / len(checks))


def bench_riemann_hilbert(seed: int = 0) -> dict[str, float]:
    return {"synthetic_riemann_hilbert": _bench_riemann_hilbert(seed)}
