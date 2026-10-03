"""Projective morphisms and Segre/Veronese embeddings (SYNTHETIC)."""

from __future__ import annotations


def segre_dim(a: int, b: int) -> int:
    """Segre P^a x P^b -> P^{(a+1)(b+1)-1}."""
    return (a + 1) * (b + 1) - 1


def veronese_degree(n: int, d: int) -> int:
    """Degree of the d-Veronese image of P^n = d^n."""
    return d**n


def veronese_target(n: int, d: int) -> int:
    """v_d: P^n -> P^{C(n+d, d) - 1}."""
    from math import comb

    return int(comb(n + d, d)) - 1


def _bench_proj_morph(seed: int = 0) -> float:
    checks = []
    # Segre P1 x P1 -> P3 (quadric surface)
    checks.append(segre_dim(1, 1) == 3)
    checks.append(segre_dim(1, 2) == 5)
    # quadric Veronese P1 -> P2 (conic): target dim 2, degree 2
    checks.append(veronese_target(1, 2) == 2)
    checks.append(veronese_degree(1, 2) == 2)
    # degree of Veronese surface v_2(P2) in P5 is 4
    checks.append(veronese_target(2, 2) == 5)
    checks.append(veronese_degree(2, 2) == 4)
    # twisted cubic v_3(P1) in P3: degree 3
    checks.append(veronese_target(1, 3) == 3)
    checks.append(veronese_degree(1, 3) == 3)
    return float(sum(checks) / len(checks))


def bench_proj_morph(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proj_morph": _bench_proj_morph(seed)}
