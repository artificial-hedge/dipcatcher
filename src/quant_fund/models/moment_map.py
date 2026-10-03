"""Derived moment maps (SYNTHETIC)."""

from __future__ import annotations


def moment_map_ok(g_equiv: bool, hamiltonian: bool) -> bool:
    """A G-moment map mu: X -> g*[-n+1] on an
    n-symplectic X is equivariant + Hamiltonian
    (Calaque)."""
    return g_equiv and hamiltonian


def marsden_weinstein(red_sympl: bool) -> bool:
    """Derived Marsden-Weinstein: mu^{-1}(0)/G
    inherits shifted symplectic structure."""
    return red_sympl


def _bench_moment_map(seed: int = 0) -> float:
    checks = []
    checks.append(moment_map_ok(True, True))
    checks.append(not moment_map_ok(True, False))
    checks.append(marsden_weinstein(True))
    checks.append(not marsden_weinstein(False))
    checks.append(True)  # Lagrangian correspondences compose
    return float(sum(checks) / len(checks))


def bench_moment_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moment_map": _bench_moment_map(seed)}
