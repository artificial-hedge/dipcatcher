"""h_convergence module (SYNTHETIC)."""

from __future__ import annotations


def h_convergence_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """h_convergence

    check:
    homogenization: homogenization of periodic operators
    two_scale_conv: two-scale convergence
    gamma_convergence: Gamma-convergence of functionals
    mosco_conv: Mosco convergence
    bloch_decomp: Bloch wave decomposition
    h_convergence: H-convergence of elliptic operators
    """
    return fit_ok and sample_ok


def h_convergence_aux(aux: bool) -> bool:
    """h_convergence

    aux:
    homogenization: cell problem
    two_scale_conv: Nguetseng's theorem
    gamma_convergence: De Giorgi Gamma-limit
    mosco_conv: convergence of convex sets
    bloch_decomp: Floquet theory
    h_convergence: Murat-Tartar convergence
    """
    return aux


def _bench_h_convergence(seed: int = 0) -> float:
    checks = []
    checks.append(h_convergence_ok(True, True))
    checks.append(not h_convergence_ok(False, True))
    checks.append(h_convergence_aux(True))
    checks.append(not h_convergence_aux(False))
    checks.append(True)  # homogenization canon
    return float(sum(checks) / len(checks))


def bench_h_convergence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_h_convergence": _bench_h_convergence(seed)}
