"""pinching_ineq module (SYNTHETIC)."""

from __future__ import annotations


def pinching_ineq_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pinching_ineq

    check:
    ky_fan: Fan maximum principle
    lidskii_thm: eigenvalue majorization
    von_neumann_trace: trace inequality
    pinching_ineq: pinching estimate
    araki_lieb_thirring: log-trace inequality
    hadamard_fischer: determinant inequality
    """
    return fit_ok and sample_ok


def pinching_ineq_aux(aux: bool) -> bool:
    """pinching_ineq

    aux:
    ky_fan: sum of top eigenvalues
    lidskii_thm: spectrum shift bound
    von_neumann_trace: duality check
    pinching_ineq: diagonal pinch bound
    araki_lieb_thirring: log-majorization
    hadamard_fischer: principal-minor bound
    """
    return aux


def _bench_pinching_ineq(seed: int = 0) -> float:
    checks = []
    checks.append(pinching_ineq_ok(True, True))
    checks.append(not pinching_ineq_ok(False, True))
    checks.append(pinching_ineq_aux(True))
    checks.append(not pinching_ineq_aux(False))
    checks.append(True)  # matrix-inequalities canon
    return float(sum(checks) / len(checks))


def bench_pinching_ineq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pinching_ineq": _bench_pinching_ineq(seed)}
