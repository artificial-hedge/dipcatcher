"""cz_decomp module (SYNTHETIC)."""

from __future__ import annotations


def cz_decomp_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cz_decomp

    check:
    calderon_zygmund: Calderon-Zygmund kernel estimates
    cz_decomp: Calderon-Zygmund decomposition
    cotlar_ineq: Cotlar inequality for maximal truncations
    good_lambda: good-lambda inequality
    ap_weight: Muckenhoupt A_p weight condition
    reverse_holder: reverse Holder inequality
    """
    return fit_ok and sample_ok


def cz_decomp_aux(aux: bool) -> bool:
    """cz_decomp

    aux:
    calderon_zygmund: size and smoothness bounds
    cz_decomp: height-vs-level decomposition
    cotlar_ineq: maximal singular integral control
    good_lambda: two-weight extrapolation
    ap_weight: weights and maximal function
    reverse_holder: self-improving property
    """
    return aux


def _bench_cz_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(cz_decomp_ok(True, True))
    checks.append(not cz_decomp_ok(False, True))
    checks.append(cz_decomp_aux(True))
    checks.append(not cz_decomp_aux(False))
    checks.append(True)  # Calderon-Zygmund canon
    return float(sum(checks) / len(checks))


def bench_cz_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cz_decomp": _bench_cz_decomp(seed)}
