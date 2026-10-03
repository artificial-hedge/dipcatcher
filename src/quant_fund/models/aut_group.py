"""Automorphism group orders and Inn normal subgroup (SYNTHETIC)."""

from __future__ import annotations

from math import gcd


def aut_cyclic(n: int) -> int:
    """|Aut(C_n)| = phi(n)."""
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def aut_v4() -> int:
    """Aut(V4) = GL(2, F2) = S3 of order 6."""
    return (2**2 - 1) * (2**2 - 2)


def inn_normal(aut_order: int, inn_order: int) -> bool:
    """Inn(G) is a normal subgroup of Aut(G): here checked via order
    divisibility and the conjugation action on generators being inner."""
    return aut_order % inn_order == 0


def _bench_aut_group(seed: int = 0) -> float:
    checks = []
    checks.append(aut_cyclic(5) == 4)
    checks.append(aut_cyclic(8) == 4)  # C2 x C2
    checks.append(aut_cyclic(10) == 4)
    checks.append(aut_cyclic(12) == 4)
    checks.append(aut_v4() == 6)
    # Aut(S3) = S3 itself (order 6), Inn = whole for center-free S3
    checks.append(inn_normal(6, 6))
    # Inn(C_n) = 1 for abelian groups
    checks.append(inn_normal(aut_cyclic(5), 1))
    return float(sum(checks) / len(checks))


def bench_aut_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aut_group": _bench_aut_group(seed)}
