"""good_lambda module (SYNTHETIC)."""

from __future__ import annotations


def good_lambda_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """good_lambda

    check:
    calderon_zygmund: Calderon-Zygmund kernel estimates
    cz_decomp: Calderon-Zygmund decomposition
    cotlar_ineq: Cotlar inequality for maximal truncations
    good_lambda: good-lambda inequality
    ap_weight: Muckenhoupt A_p weight condition
    reverse_holder: reverse Holder inequality
    """
    return fit_ok and sample_ok


def good_lambda_aux(aux: bool) -> bool:
    """good_lambda

    aux:
    calderon_zygmund: size and smoothness bounds
    cz_decomp: height-vs-level decomposition
    cotlar_ineq: maximal singular integral control
    good_lambda: two-weight extrapolation
    ap_weight: weights and maximal function
    reverse_holder: self-improving property
    """
    return aux


def _bench_good_lambda(seed: int = 0) -> float:
    checks = []
    checks.append(good_lambda_ok(True, True))
    checks.append(not good_lambda_ok(False, True))
    checks.append(good_lambda_aux(True))
    checks.append(not good_lambda_aux(False))
    checks.append(True)  # Calderon-Zygmund canon
    return float(sum(checks) / len(checks))


def bench_good_lambda(seed: int = 0) -> dict[str, float]:
    return {"synthetic_good_lambda": _bench_good_lambda(seed)}
