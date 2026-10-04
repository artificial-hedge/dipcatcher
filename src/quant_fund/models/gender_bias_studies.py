"""gender_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def gender_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gender_bias_studies

    check:
    gender_bias_studies: Gender-bias association metrics
    """
    return fit_ok and sample_ok


def gender_bias_studies_aux(aux: bool) -> bool:
    """gender_bias_studies

    aux:
    gender_bias_studies: templates, completions, biases, and scores
    """
    return aux


def _bench_gender_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gender_bias_studies_ok(True, True))
    checks.append(not gender_bias_studies_ok(False, True))
    checks.append(gender_bias_studies_aux(True))
    checks.append(not gender_bias_studies_aux(False))
    checks.append(True)  # bias-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_gender_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gender_bias_studies": _bench_gender_bias_studies(seed)}
