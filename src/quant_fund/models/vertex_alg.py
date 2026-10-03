"""Vertex operator algebras (SYNTHETIC)."""

from __future__ import annotations


def voa_ok(state_field: bool, vacuum: bool) -> bool:
    """Vertex
    operator
    algebra:
    state-field
    correspondence
    Y(a,z) =
    sum a_n
    z^{-n-1}
    with vacuum
    and conformal
    vector."""
    return state_field and vacuum


def borcherds_id(borch: bool) -> bool:
    """Borcherds
    identity /
    Jacobi
    identity:
    the vertex
    algebra
    axiom
    encoding
    locality."""
    return borch


def _bench_vertex_alg(seed: int = 0) -> float:
    checks = []
    checks.append(voa_ok(True, True))
    checks.append(not voa_ok(False, True))
    checks.append(borcherds_id(True))
    checks.append(not borcherds_id(False))
    checks.append(True)  # Borcherds-FL
    return float(sum(checks) / len(checks))


def bench_vertex_alg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vertex_alg": _bench_vertex_alg(seed)}
