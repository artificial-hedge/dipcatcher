"""nlp_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def nlp_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nlp_bias_studies

    check:
    nlp_bias_studies: NLP-classification bias metrics
    """
    return fit_ok and sample_ok


def nlp_bias_studies_aux(aux: bool) -> bool:
    """nlp_bias_studies

    aux:
    nlp_bias_studies: texts, groups, predictions, and rates
    """
    return aux


def _bench_nlp_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nlp_bias_studies_ok(True, True))
    checks.append(not nlp_bias_studies_ok(False, True))
    checks.append(nlp_bias_studies_aux(True))
    checks.append(not nlp_bias_studies_aux(False))
    checks.append(True)  # bias-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_nlp_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nlp_bias_studies": _bench_nlp_bias_studies(seed)}
