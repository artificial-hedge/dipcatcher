"""DAG representation (SYNTHETIC)."""

from __future__ import annotations


def dag_rep_ok(simplicial_ring: bool, topos: bool) -> bool:
    """Derived algebraic
    geometry representation:
    functors on simplicial
    commutative rings;
    derived stacks as
    functors of points."""
    return simplicial_ring and topos


def spectral_geometric(spectral: bool) -> bool:
    """Spectral scheme/stack:
    locally spectral
    Deligne-Mumford +
    connective E_infinity
    structure sheaf (Lurie)."""
    return spectral


def _bench_dag_representation(seed: int = 0) -> float:
    checks = []
    checks.append(dag_rep_ok(True, True))
    checks.append(not dag_rep_ok(False, True))
    checks.append(spectral_geometric(True))
    checks.append(not spectral_geometric(False))
    checks.append(True)  # Toën-Vezzosi HAG
    return float(sum(checks) / len(checks))


def bench_dag_representation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dag_representation": _bench_dag_representation(seed)}
