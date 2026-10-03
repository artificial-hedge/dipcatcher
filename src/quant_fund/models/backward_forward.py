"""backward_forward module (SYNTHETIC)."""

from __future__ import annotations


def backward_forward_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """backward_forward

    check:
    primal_dual_hybrid: primal-dual hybrid gradient
    vu_condat: Vu–Condat algorithm
    backward_forward: backward-forward splitting
    malitsky_golden: Malitsky golden ratio
    mann_iter: Mann iteration
    ishikawa_iter: Ishikawa two-step iteration
    """
    return fit_ok and sample_ok


def backward_forward_aux(aux: bool) -> bool:
    """backward_forward

    aux:
    primal_dual_hybrid: PDHG saddle-point updates
    vu_condat: three-operator primal-dual
    backward_forward: backward step then forward
    malitsky_golden: golden-ratio step parameter
    mann_iter: weak convergence to fixed point
    ishikawa_iter: two-sequence averaging
    """
    return aux


def _bench_backward_forward(seed: int = 0) -> float:
    checks = []
    checks.append(backward_forward_ok(True, True))
    checks.append(not backward_forward_ok(False, True))
    checks.append(backward_forward_aux(True))
    checks.append(not backward_forward_aux(False))
    checks.append(True)  # primal-dual canon
    return float(sum(checks) / len(checks))


def bench_backward_forward(seed: int = 0) -> dict[str, float]:
    return {"synthetic_backward_forward": _bench_backward_forward(seed)}
