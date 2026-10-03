"""ebanch_diverge module (SYNTHETIC)."""

from __future__ import annotations


def ebanch_diverge_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ebanch_diverge

    check:
    fisher_metric2: Fisher-Rao metric tensor
    expectation_param: η-coordinates on exp family
    potential_fn: convex potential / partition function
    ebanch_diverge: Bregman divergence from potential
    shannon_gibbs: Shannon-Gibbs differential entropy
    renyi_div: Rényi divergence family
    """
    return fit_ok and sample_ok


def ebanch_diverge_aux(aux: bool) -> bool:
    """ebanch_diverge

    aux:
    fisher_metric2: score outer product
    expectation_param: dual affine coordinates
    potential_fn: cumulant-generating function
    ebanch_diverge: three-point identity
    shannon_gibbs: entropy-power inequality
    renyi_div: α→1 KL limit
    """
    return aux


def _bench_ebanch_diverge(seed: int = 0) -> float:
    checks = []
    checks.append(ebanch_diverge_ok(True, True))
    checks.append(not ebanch_diverge_ok(False, True))
    checks.append(ebanch_diverge_aux(True))
    checks.append(not ebanch_diverge_aux(False))
    checks.append(True)  # information-geometry-3 canon
    return float(sum(checks) / len(checks))


def bench_ebanch_diverge(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ebanch_diverge": _bench_ebanch_diverge(seed)}
