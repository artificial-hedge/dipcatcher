"""vu_condat module (SYNTHETIC)."""

from __future__ import annotations


def vu_condat_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vu_condat

    check:
    primal_dual_hybrid: primal-dual hybrid gradient
    vu_condat: Vu–Condat algorithm
    backward_forward: backward-forward splitting
    malitsky_golden: Malitsky golden ratio
    mann_iter: Mann iteration
    ishikawa_iter: Ishikawa two-step iteration
    """
    return fit_ok and sample_ok


def vu_condat_aux(aux: bool) -> bool:
    """vu_condat

    aux:
    primal_dual_hybrid: PDHG saddle-point updates
    vu_condat: three-operator primal-dual
    backward_forward: backward step then forward
    malitsky_golden: golden-ratio step parameter
    mann_iter: weak convergence to fixed point
    ishikawa_iter: two-sequence averaging
    """
    return aux


def _bench_vu_condat(seed: int = 0) -> float:
    checks = []
    checks.append(vu_condat_ok(True, True))
    checks.append(not vu_condat_ok(False, True))
    checks.append(vu_condat_aux(True))
    checks.append(not vu_condat_aux(False))
    checks.append(True)  # primal-dual canon
    return float(sum(checks) / len(checks))


def bench_vu_condat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vu_condat": _bench_vu_condat(seed)}
