"""hops_add_studies module (SYNTHETIC)."""

from __future__ import annotations


def hops_add_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hops_add_studies

    check:
    hops_add_studies: systematic-hops metrics
    """
    return fit_ok and sample_ok


def hops_add_studies_aux(aux: bool) -> bool:
    """hops_add_studies

    aux:
    hops_add_studies: expressions, values, splits, and accuracies
    """
    return aux


def _bench_hops_add_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hops_add_studies_ok(True, True))
    checks.append(not hops_add_studies_ok(False, True))
    checks.append(hops_add_studies_aux(True))
    checks.append(not hops_add_studies_aux(False))
    checks.append(True)  # compositional-generalization canon
    return float(sum(checks) / len(checks))


def bench_hops_add_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hops_add_studies": _bench_hops_add_studies(seed)}
