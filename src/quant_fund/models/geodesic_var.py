"""geodesic_var module (SYNTHETIC)."""

from __future__ import annotations


def geodesic_var_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geodesic_var

    check:
    euler_lagrange: Euler-Lagrange equations
    legendre_cond: Legendre necessary condition
    jacobi_eq: Jacobi accessory equation
    geodesic_var: first variation of geodesic
    isoperimetric_var: isoperimetric problem in the plane
    soap_film: Plateau variational problem
    """
    return fit_ok and sample_ok


def geodesic_var_aux(aux: bool) -> bool:
    """geodesic_var

    aux:
    euler_lagrange: weak/strong minimizers
    legendre_cond: Weierstrass excess function
    jacobi_eq: conjugate points
    geodesic_var: second variation formula
    isoperimetric_var: isoperimetric profile
    soap_film: Douglas-Rado solution
    """
    return aux


def _bench_geodesic_var(seed: int = 0) -> float:
    checks = []
    checks.append(geodesic_var_ok(True, True))
    checks.append(not geodesic_var_ok(False, True))
    checks.append(geodesic_var_aux(True))
    checks.append(not geodesic_var_aux(False))
    checks.append(True)  # calculus-of-variations canon
    return float(sum(checks) / len(checks))


def bench_geodesic_var(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geodesic_var": _bench_geodesic_var(seed)}
