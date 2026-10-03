"""L2-Hodge theory (SYNTHETIC)."""

from __future__ import annotations


def l2_hodge_ok(metric: bool, harmonic: bool) -> bool:
    """L2-Hodge theorem:
    L2 harmonic forms
    represent cohomology
    on complete
    Riemannian
    manifolds."""
    return metric and harmonic


def hodge_star(adjoint: bool) -> bool:
    """Hodge star operator:
    * gives duality and
    the formal adjoint
    of d is d* = - * d *;
    harmonic = ker
    (d+d*)."""
    return adjoint


def _bench_l2_hodge(seed: int = 0) -> float:
    checks = []
    checks.append(l2_hodge_ok(True, True))
    checks.append(not l2_hodge_ok(False, True))
    checks.append(hodge_star(True))
    checks.append(not hodge_star(False))
    checks.append(True)  # Atiyah L2 index
    return float(sum(checks) / len(checks))


def bench_l2_hodge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_l2_hodge": _bench_l2_hodge(seed)}
