"""Cap product on manifolds (SYNTHETIC)."""

from __future__ import annotations


def cap(h_cohom_dim: int, h_hom_dim: int, manifold_dim: int) -> int | None:
    """H^p x H_n -> H_{n-p}: defined when 0 <= p <= n; result in
    dimension n-p, valid only if both groups nonzero on a manifold
    with homology only in {0, d}."""
    out = h_hom_dim - h_cohom_dim
    if out < 0 or h_cohom_dim < 0:
        return None
    if h_cohom_dim not in (0, manifold_dim):
        return 0
    if h_hom_dim not in (0, manifold_dim):
        return 0
    return out


def _bench_cap_product(seed: int = 0) -> float:
    checks = []
    # On S^3: H^3 cap H_3 -> H_0 (generator maps to point class)
    checks.append(cap(3, 3, 3) == 0)
    # H^0 cap H_3 = H_3 (unit acts as identity)
    checks.append(cap(0, 3, 3) == 3)
    # cap is zero in the sparse homology of S^3 for middle degrees
    checks.append(cap(1, 3, 3) == 0)
    # negative-dimension output is impossible
    checks.append(cap(3, 0, 3) is None)
    # unit cap on H_0 returns H_0
    checks.append(cap(0, 0, 3) == 0)
    return float(sum(checks) / len(checks))


def bench_cap_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cap_product": _bench_cap_product(seed)}
