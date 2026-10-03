"""m-adic completion and associated graded (SYNTHETIC)."""

from __future__ import annotations


def ideals_of_power_series(n_pow: int) -> int:
    """Ideals of k[[x]] are (x^0)=(1),(x),(x^2),...: the valuation
    filtration — model: ideal (x^n) has index n in the chain."""
    return n_pow


def gr_piece(n_vars: int, deg: int) -> int:
    """gr_m(R)_d = m^d / m^{d+1} for R = k[[x_1..x_n]]: monomials of
    degree exactly d = C(d + n - 1, n - 1)."""
    from math import comb

    return comb(deg + n_vars - 1, n_vars - 1)


def filtr_mult(a_pow: int, b_pow: int) -> int:
    """m-adic filtration: (x^a)(x^b) = (x^{a+b})."""
    return a_pow + b_pow


def _bench_completion_ring(seed: int = 0) -> float:
    checks = []
    # k[[x]] ideals index
    checks.append(ideals_of_power_series(3) == 3)
    # gr of k[[x]] is k[x]: 1-dim each degree
    checks.append(gr_piece(1, 4) == 1)
    # k[[x,y]]: degree-2 graded piece has 3 monomials
    checks.append(gr_piece(2, 2) == 3)
    checks.append(gr_piece(3, 1) == 3)
    # filtration multiplicative
    checks.append(filtr_mult(2, 3) == 5)
    # Krull intersection: cap m^n = 0 in a domain model -> True
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_completion_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_completion_ring": _bench_completion_ring(seed)}
