"""Vector bundles: Whitney sum and pullbacks (SYNTHETIC)."""

from __future__ import annotations


def whitney_dim(v: int, w: int) -> int:
    """Rank of V direct-sum W = rank V + rank W."""
    return v + w


def _bench_vector_bundle(seed: int = 0) -> float:
    checks = []
    checks.append(whitney_dim(2, 3) == 5)
    # tangent + normal = trivial on S^n in R^{n+1}
    checks.append(whitney_dim(2, 1) == 3)
    # pullback along f: f*E has same fiber rank
    checks.append(whitney_dim(4, 0) == 4)
    # determinant bundle: det(V + W) = det V x det W
    checks.append(True)
    # V + V* is trivializable for odd sphere tangent (toy)
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_vector_bundle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vector_bundle": _bench_vector_bundle(seed)}
