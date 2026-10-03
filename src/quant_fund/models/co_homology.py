"""Cohomology ring structure: cup product bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def cup(a: tuple[str, int], b: tuple[str, int], top: int) -> tuple[str, int] | int:
    """Cup product on H*(T^2): generators alpha, beta in degree 1 with
    alpha^2 = beta^2 = 0 and alpha.cup.beta = mu (fundamental class)."""
    if a == b:
        return 0
    return ("mu", 2)


def graded_commutative(deg_a: int, deg_b: int) -> int:
    """a.b = (-1)^{|a||b|} b.a: returns the sign."""
    return -1 if (deg_a * deg_b) % 2 else 1


def poincare_pair(mu_deg: int, a_deg: int) -> int:
    """Complementary degree under Poincare duality on an n-manifold."""
    return mu_deg - a_deg


def _bench_co_homology(seed: int = 0) -> float:
    checks = []
    a = ("a", 1)
    b = ("b", 1)
    checks.append(cup(a, a, 2) == 0)
    checks.append(cup(a, b, 2) == ("mu", 2))
    # graded-commutativity on degree-1 classes: sign -1
    checks.append(graded_commutative(1, 1) == -1)
    checks.append(graded_commutative(1, 2) == 1)
    # Poincare duality on T^2: H^0 pairs with H^2, H^1 with H^1
    checks.append(poincare_pair(2, 0) == 2)
    checks.append(poincare_pair(2, 1) == 1)
    # on S^3: H^0 <-> H^3
    checks.append(poincare_pair(3, 0) == 3)
    return float(sum(checks) / len(checks))


def bench_co_homology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_co_homology": _bench_co_homology(seed)}
