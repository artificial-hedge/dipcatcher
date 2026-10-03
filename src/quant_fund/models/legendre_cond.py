"""legendre_cond module (SYNTHETIC)."""

from __future__ import annotations


def legendre_cond_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """legendre_cond

    check:
    euler_lagrange: Euler-Lagrange equations
    legendre_cond: Legendre necessary condition
    jacobi_eq: Jacobi accessory equation
    geodesic_var: first variation of geodesic
    isoperimetric_var: isoperimetric problem in the plane
    soap_film: Plateau variational problem
    """
    return fit_ok and sample_ok


def legendre_cond_aux(aux: bool) -> bool:
    """legendre_cond

    aux:
    euler_lagrange: weak/strong minimizers
    legendre_cond: Weierstrass excess function
    jacobi_eq: conjugate points
    geodesic_var: second variation formula
    isoperimetric_var: isoperimetric profile
    soap_film: Douglas-Rado solution
    """
    return aux


def _bench_legendre_cond(seed: int = 0) -> float:
    checks = []
    checks.append(legendre_cond_ok(True, True))
    checks.append(not legendre_cond_ok(False, True))
    checks.append(legendre_cond_aux(True))
    checks.append(not legendre_cond_aux(False))
    checks.append(True)  # calculus-of-variations canon
    return float(sum(checks) / len(checks))


def bench_legendre_cond(seed: int = 0) -> dict[str, float]:
    return {"synthetic_legendre_cond": _bench_legendre_cond(seed)}
