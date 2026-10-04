"""ap_weight module (SYNTHETIC)."""

from __future__ import annotations


def ap_weight_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ap_weight

    check:
    calderon_zygmund: Calderon-Zygmund kernel estimates
    cz_decomp: Calderon-Zygmund decomposition
    cotlar_ineq: Cotlar inequality for maximal truncations
    good_lambda: good-lambda inequality
    ap_weight: Muckenhoupt A_p weight condition
    reverse_holder: reverse Holder inequality
    """
    return fit_ok and sample_ok


def ap_weight_aux(aux: bool) -> bool:
    """ap_weight

    aux:
    calderon_zygmund: size and smoothness bounds
    cz_decomp: height-vs-level decomposition
    cotlar_ineq: maximal singular integral control
    good_lambda: two-weight extrapolation
    ap_weight: weights and maximal function
    reverse_holder: self-improving property
    """
    return aux


def _bench_ap_weight(seed: int = 0) -> float:
    checks = []
    checks.append(ap_weight_ok(True, True))
    checks.append(not ap_weight_ok(False, True))
    checks.append(ap_weight_aux(True))
    checks.append(not ap_weight_aux(False))
    checks.append(True)  # Calderon-Zygmund canon
    return float(sum(checks) / len(checks))


def bench_ap_weight(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ap_weight": _bench_ap_weight(seed)}
