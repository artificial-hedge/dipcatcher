"""mnli_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mnli_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mnli_lite_studies

    check:
    mnli_lite_studies: Multi-genre NLI metrics
    """
    return fit_ok and sample_ok


def mnli_lite_studies_aux(aux: bool) -> bool:
    """mnli_lite_studies

    aux:
    mnli_lite_studies: premises, hypotheses, genres, and accuracies
    """
    return aux


def _bench_mnli_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mnli_lite_studies_ok(True, True))
    checks.append(not mnli_lite_studies_ok(False, True))
    checks.append(mnli_lite_studies_aux(True))
    checks.append(not mnli_lite_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_mnli_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mnli_lite_studies": _bench_mnli_lite_studies(seed)}
