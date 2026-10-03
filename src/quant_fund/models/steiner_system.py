"""Steiner triple / BIBD parameters (SYNTHETIC)."""

from __future__ import annotations

import math


def bibd_b(v: int, k: int, lam: int) -> int:
    """b = lam * C(v,2) / C(k,2) for a 2-(v,k,lam) design."""
    return lam * math.comb(v, 2) // math.comb(k, 2)


def bibd_r(v: int, k: int, lam: int) -> int:
    """r = lam * (v-1) / (k-1)."""
    return lam * (v - 1) // (k - 1)


def is_admissible(v: int, k: int, lam: int) -> bool:
    """Integrality conditions: r and b must be integers."""
    return lam * (v - 1) % (k - 1) == 0 and lam * v * (v - 1) % (k * (k - 1)) == 0


def _bench_steiner_system(seed: int = 0) -> float:
    checks = []
    # Fano plane STS(7): b = 7, r = 3
    checks.append(bibd_b(7, 3, 1) == 7)
    checks.append(bibd_r(7, 3, 1) == 3)
    # affine plane of order 3 = STS(9): b = 12, r = 4
    checks.append(bibd_b(9, 3, 1) == 12)
    checks.append(bibd_r(9, 3, 1) == 4)
    # STS exists iff v == 1 or 3 mod 6: v=7 ok, v=8 not
    checks.append(is_admissible(7, 3, 1))
    checks.append(not is_admissible(8, 3, 1))
    # Fisher: b >= v for nontrivial BIBD — check Fano satisfies
    checks.append(bibd_b(7, 3, 1) >= 7)
    return float(sum(checks) / len(checks))


def bench_steiner_system(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steiner_system": _bench_steiner_system(seed)}
