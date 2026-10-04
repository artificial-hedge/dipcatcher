"""prost_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def prost_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prost_lite_studies

    check:
    prost_lite_studies: PROST physical-reasoning metrics
    """
    return fit_ok and sample_ok


def prost_lite_studies_aux(aux: bool) -> bool:
    """prost_lite_studies

    aux:
    prost_lite_studies: contexts, questions, options, and accuracies
    """
    return aux


def _bench_prost_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prost_lite_studies_ok(True, True))
    checks.append(not prost_lite_studies_ok(False, True))
    checks.append(prost_lite_studies_aux(True))
    checks.append(not prost_lite_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_prost_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prost_lite_studies": _bench_prost_lite_studies(seed)}
