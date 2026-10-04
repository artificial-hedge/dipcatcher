"""anli_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def anli_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anli_lite_studies

    check:
    anli_lite_studies: Adversarial NLI metrics
    """
    return fit_ok and sample_ok


def anli_lite_studies_aux(aux: bool) -> bool:
    """anli_lite_studies

    aux:
    anli_lite_studies: premises, hypotheses, labels, and accuracies
    """
    return aux


def _bench_anli_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anli_lite_studies_ok(True, True))
    checks.append(not anli_lite_studies_ok(False, True))
    checks.append(anli_lite_studies_aux(True))
    checks.append(not anli_lite_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_anli_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anli_lite_studies": _bench_anli_lite_studies(seed)}
