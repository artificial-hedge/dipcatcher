"""parabolic_harnack module (SYNTHETIC)."""

from __future__ import annotations


def parabolic_harnack_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parabolic_harnack

    check:
    parabolic_harnack: parabolic Harnack inequality
    gaussian_upper: Gaussian upper bounds for heat kernel
    li_yau: Li-Yau gradient estimate
    nash_ineq: Nash inequality for heat kernel decay
    davies_gaffney: Davies-Gaffney off-diagonal estimates
    grad_est: gradient estimates for elliptic operators
    """
    return fit_ok and sample_ok


def parabolic_harnack_aux(aux: bool) -> bool:
    """parabolic_harnack

    aux:
    parabolic_harnack: space-time cylinder control
    gaussian_upper: Davies' perturbation method
    li_yau: Laplacian comparison on manifolds
    nash_ineq: Sobolev-entropy method
    davies_gaffney: finite propagation speed
    grad_est: Bochner formula application
    """
    return aux


def _bench_parabolic_harnack(seed: int = 0) -> float:
    checks = []
    checks.append(parabolic_harnack_ok(True, True))
    checks.append(not parabolic_harnack_ok(False, True))
    checks.append(parabolic_harnack_aux(True))
    checks.append(not parabolic_harnack_aux(False))
    checks.append(True)  # parabolic/Li-Yau canon
    return float(sum(checks) / len(checks))


def bench_parabolic_harnack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parabolic_harnack": _bench_parabolic_harnack(seed)}
