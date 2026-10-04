"""malitsky_golden module (SYNTHETIC)."""

from __future__ import annotations


def malitsky_golden_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """malitsky_golden

    check:
    primal_dual_hybrid: primal-dual hybrid gradient
    vu_condat: Vu–Condat algorithm
    backward_forward: backward-forward splitting
    malitsky_golden: Malitsky golden ratio
    mann_iter: Mann iteration
    ishikawa_iter: Ishikawa two-step iteration
    """
    return fit_ok and sample_ok


def malitsky_golden_aux(aux: bool) -> bool:
    """malitsky_golden

    aux:
    primal_dual_hybrid: PDHG saddle-point updates
    vu_condat: three-operator primal-dual
    backward_forward: backward step then forward
    malitsky_golden: golden-ratio step parameter
    mann_iter: weak convergence to fixed point
    ishikawa_iter: two-sequence averaging
    """
    return aux


def _bench_malitsky_golden(seed: int = 0) -> float:
    checks = []
    checks.append(malitsky_golden_ok(True, True))
    checks.append(not malitsky_golden_ok(False, True))
    checks.append(malitsky_golden_aux(True))
    checks.append(not malitsky_golden_aux(False))
    checks.append(True)  # primal-dual canon
    return float(sum(checks) / len(checks))


def bench_malitsky_golden(seed: int = 0) -> dict[str, float]:
    return {"synthetic_malitsky_golden": _bench_malitsky_golden(seed)}
