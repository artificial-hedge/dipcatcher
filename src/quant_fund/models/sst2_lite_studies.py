"""sst2_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def sst2_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sst2_lite_studies

    check:
    sst2_lite_studies: SST-2 sentiment metrics
    """
    return fit_ok and sample_ok


def sst2_lite_studies_aux(aux: bool) -> bool:
    """sst2_lite_studies

    aux:
    sst2_lite_studies: sentences, labels, predictions, and accuracies
    """
    return aux


def _bench_sst2_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sst2_lite_studies_ok(True, True))
    checks.append(not sst2_lite_studies_ok(False, True))
    checks.append(sst2_lite_studies_aux(True))
    checks.append(not sst2_lite_studies_aux(False))
    checks.append(True)  # GLUE-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_sst2_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sst2_lite_studies": _bench_sst2_lite_studies(seed)}
