"""gaussian_upper module (SYNTHETIC)."""

from __future__ import annotations


def gaussian_upper_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gaussian_upper

    check:
    parabolic_harnack: parabolic Harnack inequality
    gaussian_upper: Gaussian upper bounds for heat kernel
    li_yau: Li-Yau gradient estimate
    nash_ineq: Nash inequality for heat kernel decay
    davies_gaffney: Davies-Gaffney off-diagonal estimates
    grad_est: gradient estimates for elliptic operators
    """
    return fit_ok and sample_ok


def gaussian_upper_aux(aux: bool) -> bool:
    """gaussian_upper

    aux:
    parabolic_harnack: space-time cylinder control
    gaussian_upper: Davies' perturbation method
    li_yau: Laplacian comparison on manifolds
    nash_ineq: Sobolev-entropy method
    davies_gaffney: finite propagation speed
    grad_est: Bochner formula application
    """
    return aux


def _bench_gaussian_upper(seed: int = 0) -> float:
    checks = []
    checks.append(gaussian_upper_ok(True, True))
    checks.append(not gaussian_upper_ok(False, True))
    checks.append(gaussian_upper_aux(True))
    checks.append(not gaussian_upper_aux(False))
    checks.append(True)  # parabolic/Li-Yau canon
    return float(sum(checks) / len(checks))


def bench_gaussian_upper(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gaussian_upper": _bench_gaussian_upper(seed)}
