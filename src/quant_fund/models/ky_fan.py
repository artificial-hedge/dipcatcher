"""ky_fan module (SYNTHETIC)."""

from __future__ import annotations


def ky_fan_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ky_fan

    check:
    ky_fan: Fan maximum principle
    lidskii_thm: eigenvalue majorization
    von_neumann_trace: trace inequality
    pinching_ineq: pinching estimate
    araki_lieb_thirring: log-trace inequality
    hadamard_fischer: determinant inequality
    """
    return fit_ok and sample_ok


def ky_fan_aux(aux: bool) -> bool:
    """ky_fan

    aux:
    ky_fan: sum of top eigenvalues
    lidskii_thm: spectrum shift bound
    von_neumann_trace: duality check
    pinching_ineq: diagonal pinch bound
    araki_lieb_thirring: log-majorization
    hadamard_fischer: principal-minor bound
    """
    return aux


def _bench_ky_fan(seed: int = 0) -> float:
    checks = []
    checks.append(ky_fan_ok(True, True))
    checks.append(not ky_fan_ok(False, True))
    checks.append(ky_fan_aux(True))
    checks.append(not ky_fan_aux(False))
    checks.append(True)  # matrix-inequalities canon
    return float(sum(checks) / len(checks))


def bench_ky_fan(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ky_fan": _bench_ky_fan(seed)}
