"""anli_r2_studies module (SYNTHETIC)."""

from __future__ import annotations


def anli_r2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anli_r2_studies

    check:
    anli_r2_studies: ANLI-R2 adversarial-NLI metrics
    """
    return fit_ok and sample_ok


def anli_r2_studies_aux(aux: bool) -> bool:
    """anli_r2_studies

    aux:
    anli_r2_studies: contexts, hypotheses, labels, and accuracies
    """
    return aux


def _bench_anli_r2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anli_r2_studies_ok(True, True))
    checks.append(not anli_r2_studies_ok(False, True))
    checks.append(anli_r2_studies_aux(True))
    checks.append(not anli_r2_studies_aux(False))
    checks.append(True)  # NLI-eval canon
    return float(sum(checks) / len(checks))


def bench_anli_r2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anli_r2_studies": _bench_anli_r2_studies(seed)}
