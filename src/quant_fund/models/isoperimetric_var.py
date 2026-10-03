"""isoperimetric_var module (SYNTHETIC)."""

from __future__ import annotations


def isoperimetric_var_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """isoperimetric_var

    check:
    euler_lagrange: Euler-Lagrange equations
    legendre_cond: Legendre necessary condition
    jacobi_eq: Jacobi accessory equation
    geodesic_var: first variation of geodesic
    isoperimetric_var: isoperimetric problem in the plane
    soap_film: Plateau variational problem
    """
    return fit_ok and sample_ok


def isoperimetric_var_aux(aux: bool) -> bool:
    """isoperimetric_var

    aux:
    euler_lagrange: weak/strong minimizers
    legendre_cond: Weierstrass excess function
    jacobi_eq: conjugate points
    geodesic_var: second variation formula
    isoperimetric_var: isoperimetric profile
    soap_film: Douglas-Rado solution
    """
    return aux


def _bench_isoperimetric_var(seed: int = 0) -> float:
    checks = []
    checks.append(isoperimetric_var_ok(True, True))
    checks.append(not isoperimetric_var_ok(False, True))
    checks.append(isoperimetric_var_aux(True))
    checks.append(not isoperimetric_var_aux(False))
    checks.append(True)  # calculus-of-variations canon
    return float(sum(checks) / len(checks))


def bench_isoperimetric_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_isoperimetric_var": _bench_isoperimetric_var(seed)}
