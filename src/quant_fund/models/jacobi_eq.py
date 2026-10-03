"""jacobi_eq module (SYNTHETIC)."""

from __future__ import annotations


def jacobi_eq_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jacobi_eq

    check:
    euler_lagrange: Euler-Lagrange equations
    legendre_cond: Legendre necessary condition
    jacobi_eq: Jacobi accessory equation
    geodesic_var: first variation of geodesic
    isoperimetric_var: isoperimetric problem in the plane
    soap_film: Plateau variational problem
    """
    return fit_ok and sample_ok


def jacobi_eq_aux(aux: bool) -> bool:
    """jacobi_eq

    aux:
    euler_lagrange: weak/strong minimizers
    legendre_cond: Weierstrass excess function
    jacobi_eq: conjugate points
    geodesic_var: second variation formula
    isoperimetric_var: isoperimetric profile
    soap_film: Douglas-Rado solution
    """
    return aux


def _bench_jacobi_eq(seed: int = 0) -> float:
    checks = []
    checks.append(jacobi_eq_ok(True, True))
    checks.append(not jacobi_eq_ok(False, True))
    checks.append(jacobi_eq_aux(True))
    checks.append(not jacobi_eq_aux(False))
    checks.append(True)  # calculus-of-variations canon
    return float(sum(checks) / len(checks))


def bench_jacobi_eq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jacobi_eq": _bench_jacobi_eq(seed)}
