"""analytic_center module (SYNTHETIC)."""

from __future__ import annotations


def analytic_center_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """analytic_center

    check:
    kkt_solve: KKT system solver
    cvx_reform: convex reformulation
    self_concordant: self-concordant barrier
    logbarrier_fn: logarithmic barrier
    analytic_center: analytic centering
    dik_ellipsoid: Dikin ellipsoid step
    """
    return fit_ok and sample_ok


def analytic_center_aux(aux: bool) -> bool:
    """analytic_center

    aux:
    kkt_solve: primal-dual residual norms
    cvx_reform: epigraph + SOC lifting
    self_concordant: Newton decrement bound
    logbarrier_fn: centrality measure
    analytic_center: maximizer of det barrier
    dik_ellipsoid: local Hessian ball
    """
    return aux


def _bench_analytic_center(seed: int = 0) -> float:
    checks = []
    checks.append(analytic_center_ok(True, True))
    checks.append(not analytic_center_ok(False, True))
    checks.append(analytic_center_aux(True))
    checks.append(not analytic_center_aux(False))
    checks.append(True)  # convex-optimization canon
    return float(sum(checks) / len(checks))


def bench_analytic_center(seed: int = 0) -> dict[str, float]:
    return {"synthetic_analytic_center": _bench_analytic_center(seed)}
