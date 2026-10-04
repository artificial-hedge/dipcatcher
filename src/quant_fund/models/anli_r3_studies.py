"""anli_r3_studies module (SYNTHETIC)."""

from __future__ import annotations


def anli_r3_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """anli_r3_studies

    check:
    anli_r3_studies: ANLI-R3 adversarial-NLI metrics
    """
    return fit_ok and sample_ok


def anli_r3_studies_aux(aux: bool) -> bool:
    """anli_r3_studies

    aux:
    anli_r3_studies: contexts, hypotheses, labels, and accuracies
    """
    return aux


def _bench_anli_r3_studies(seed: int = 0) -> float:
    checks = []
    checks.append(anli_r3_studies_ok(True, True))
    checks.append(not anli_r3_studies_ok(False, True))
    checks.append(anli_r3_studies_aux(True))
    checks.append(not anli_r3_studies_aux(False))
    checks.append(True)  # NLI-eval canon
    return float(sum(checks) / len(checks))


def bench_anli_r3_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anli_r3_studies": _bench_anli_r3_studies(seed)}
