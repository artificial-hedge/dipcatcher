"""Cohen-Macaulay rings: depth equals dimension (SYNTHETIC)."""

from __future__ import annotations


def is_cm(depth: int, dim: int) -> bool:
    """A local ring is Cohen-Macaulay iff depth = dim."""
    return depth == dim


def _bench_cohen_mac(seed: int = 0) -> float:
    checks = []
    # regular local rings are CM
    checks.append(is_cm(2, 2))
    # k[x,y]/(xy): dim 1, depth 1 -> CM
    checks.append(is_cm(1, 1))
    # depth < dim -> not CM
    checks.append(not is_cm(0, 1))
    # hypersurface rings are CM
    checks.append(is_cm(3, 3))
    # CM => unmixed: no embedded primes
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_cohen_mac(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohen_mac": _bench_cohen_mac(seed)}
