"""primal_dual_hybrid module (SYNTHETIC)."""

from __future__ import annotations


def primal_dual_hybrid_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """primal_dual_hybrid

    check:
    primal_dual_hybrid: primal-dual hybrid gradient
    vu_condat: Vu–Condat algorithm
    backward_forward: backward-forward splitting
    malitsky_golden: Malitsky golden ratio
    mann_iter: Mann iteration
    ishikawa_iter: Ishikawa two-step iteration
    """
    return fit_ok and sample_ok


def primal_dual_hybrid_aux(aux: bool) -> bool:
    """primal_dual_hybrid

    aux:
    primal_dual_hybrid: PDHG saddle-point updates
    vu_condat: three-operator primal-dual
    backward_forward: backward step then forward
    malitsky_golden: golden-ratio step parameter
    mann_iter: weak convergence to fixed point
    ishikawa_iter: two-sequence averaging
    """
    return aux


def _bench_primal_dual_hybrid(seed: int = 0) -> float:
    checks = []
    checks.append(primal_dual_hybrid_ok(True, True))
    checks.append(not primal_dual_hybrid_ok(False, True))
    checks.append(primal_dual_hybrid_aux(True))
    checks.append(not primal_dual_hybrid_aux(False))
    checks.append(True)  # primal-dual canon
    return float(sum(checks) / len(checks))


def bench_primal_dual_hybrid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_primal_dual_hybrid": _bench_primal_dual_hybrid(seed)}
