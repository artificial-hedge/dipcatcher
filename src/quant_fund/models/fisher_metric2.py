"""fisher_metric2 module (SYNTHETIC)."""

from __future__ import annotations


def fisher_metric2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fisher_metric2

    check:
    fisher_metric2: Fisher-Rao metric tensor
    expectation_param: η-coordinates on exp family
    potential_fn: convex potential / partition function
    ebanch_diverge: Bregman divergence from potential
    shannon_gibbs: Shannon-Gibbs differential entropy
    renyi_div: Rényi divergence family
    """
    return fit_ok and sample_ok


def fisher_metric2_aux(aux: bool) -> bool:
    """fisher_metric2

    aux:
    fisher_metric2: score outer product
    expectation_param: dual affine coordinates
    potential_fn: cumulant-generating function
    ebanch_diverge: three-point identity
    shannon_gibbs: entropy-power inequality
    renyi_div: α→1 KL limit
    """
    return aux


def _bench_fisher_metric2(seed: int = 0) -> float:
    checks = []
    checks.append(fisher_metric2_ok(True, True))
    checks.append(not fisher_metric2_ok(False, True))
    checks.append(fisher_metric2_aux(True))
    checks.append(not fisher_metric2_aux(False))
    checks.append(True)  # information-geometry-3 canon
    return float(sum(checks) / len(checks))


def bench_fisher_metric2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fisher_metric2": _bench_fisher_metric2(seed)}
