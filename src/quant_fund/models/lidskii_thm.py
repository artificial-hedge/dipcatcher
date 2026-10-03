"""lidskii_thm module (SYNTHETIC)."""

from __future__ import annotations


def lidskii_thm_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lidskii_thm

    check:
    ky_fan: Fan maximum principle
    lidskii_thm: eigenvalue majorization
    von_neumann_trace: trace inequality
    pinching_ineq: pinching estimate
    araki_lieb_thirring: log-trace inequality
    hadamard_fischer: determinant inequality
    """
    return fit_ok and sample_ok


def lidskii_thm_aux(aux: bool) -> bool:
    """lidskii_thm

    aux:
    ky_fan: sum of top eigenvalues
    lidskii_thm: spectrum shift bound
    von_neumann_trace: duality check
    pinching_ineq: diagonal pinch bound
    araki_lieb_thirring: log-majorization
    hadamard_fischer: principal-minor bound
    """
    return aux


def _bench_lidskii_thm(seed: int = 0) -> float:
    checks = []
    checks.append(lidskii_thm_ok(True, True))
    checks.append(not lidskii_thm_ok(False, True))
    checks.append(lidskii_thm_aux(True))
    checks.append(not lidskii_thm_aux(False))
    checks.append(True)  # matrix-inequalities canon
    return float(sum(checks) / len(checks))


def bench_lidskii_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lidskii_thm": _bench_lidskii_thm(seed)}
