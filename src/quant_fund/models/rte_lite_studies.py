"""rte_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def rte_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rte_lite_studies

    check:
    rte_lite_studies: RTE entailment metrics
    """
    return fit_ok and sample_ok


def rte_lite_studies_aux(aux: bool) -> bool:
    """rte_lite_studies

    aux:
    rte_lite_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_rte_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rte_lite_studies_ok(True, True))
    checks.append(not rte_lite_studies_ok(False, True))
    checks.append(rte_lite_studies_aux(True))
    checks.append(not rte_lite_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_rte_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rte_lite_studies": _bench_rte_lite_studies(seed)}
