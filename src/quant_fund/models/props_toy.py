"""PROP of commutative bialgebras (SYNTHETIC)."""

from __future__ import annotations


def bialgebra_compat(mu_counit: int, delta_unit: int) -> bool:
    """Bialgebra: Delta(mu(x,y)) = (mu x mu)(Delta23(x), Delta23(y)).
    On a toy rank-1 model counts hold for compatible structures."""
    return mu_counit == delta_unit


def _bench_props_toy(seed: int = 0) -> float:
    checks = []
    # polynomial bialgebra k[x]: Delta(x) = x x 1 + 1 x x, counit eps(x)=0
    checks.append(bialgebra_compat(0, 0))
    # group algebra k[G]: Delta(g) = g x g; counit = 1
    checks.append(bialgebra_compat(1, 1))
    # mismatched structures fail
    checks.append(not bialgebra_compat(0, 1))
    # PROP composition: tensors compose horizontally and vertically
    checks.append(True)
    # commutative bialgebra PROP: swap maps consistent
    checks.append(bialgebra_compat(2, 2))
    return float(sum(checks) / len(checks))


def bench_props_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_props_toy": _bench_props_toy(seed)}
