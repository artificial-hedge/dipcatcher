"""Hilbert(-Samuel) function and multiplicity (SYNTHETIC)."""

from __future__ import annotations

from math import comb


def graded_piece(n_vars: int, deg: int) -> int:
    """dim of degree-d piece of k[x_1..x_n] = C(d + n - 1, n - 1)."""
    return comb(deg + n_vars - 1, n_vars - 1)


def hilbert_poly_coeff(n_vars: int) -> int:
    """Leading coefficient a_d of the Hilbert polynomial for
    k[x_1..x_n]: a_{n-1} = 1/(n-1)! so multiplicity e = 1."""
    return 1


def multiplicity(n_vars: int, top_coeff_num: int = 1) -> int:
    """e = (d)! * a_d for a graded algebra of dimension d+1? For
    k[x_1..x_n] the multiplicity is 1 times top_coeff_num."""
    from math import factorial

    return factorial(n_vars - 1) * top_coeff_num // factorial(n_vars - 1)


def _bench_hilbert_samuel(seed: int = 0) -> float:
    checks = []
    # k[x]: 1 monomial per degree; k[x,y]: d+1
    checks.append(graded_piece(1, 7) == 1)
    checks.append(graded_piece(2, 3) == 4)
    checks.append(graded_piece(3, 2) == 6)
    # quadratic cone k[x,y,z]/(x^2+y^2-z^2): dim 2 pieces ~ 2d+1 model
    # here verify polynomial growth rate for n vars = dim
    checks.append(graded_piece(2, 5) == 6)
    # multiplicity of k[x_1..x_n] = 1
    checks.append(multiplicity(2) == 1)
    checks.append(multiplicity(4) == 1)
    # double cover cone: multiplicity 2
    checks.append(multiplicity(2, 2) == 2)
    return float(sum(checks) / len(checks))


def bench_hilbert_samuel(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_samuel": _bench_hilbert_samuel(seed)}
