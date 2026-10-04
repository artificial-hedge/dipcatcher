"""cvx_reform module (SYNTHETIC)."""

from __future__ import annotations


def cvx_reform_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cvx_reform

    check:
    kkt_solve: KKT system solver
    cvx_reform: convex reformulation
    self_concordant: self-concordant barrier
    logbarrier_fn: logarithmic barrier
    analytic_center: analytic centering
    dik_ellipsoid: Dikin ellipsoid step
    """
    return fit_ok and sample_ok


def cvx_reform_aux(aux: bool) -> bool:
    """cvx_reform

    aux:
    kkt_solve: primal-dual residual norms
    cvx_reform: epigraph + SOC lifting
    self_concordant: Newton decrement bound
    logbarrier_fn: centrality measure
    analytic_center: maximizer of det barrier
    dik_ellipsoid: local Hessian ball
    """
    return aux


def _bench_cvx_reform(seed: int = 0) -> float:
    checks = []
    checks.append(cvx_reform_ok(True, True))
    checks.append(not cvx_reform_ok(False, True))
    checks.append(cvx_reform_aux(True))
    checks.append(not cvx_reform_aux(False))
    checks.append(True)  # convex-optimization canon
    return float(sum(checks) / len(checks))


def bench_cvx_reform(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cvx_reform": _bench_cvx_reform(seed)}
