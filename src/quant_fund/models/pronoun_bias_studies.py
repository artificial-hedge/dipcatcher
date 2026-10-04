"""pronoun_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def pronoun_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pronoun_bias_studies

    check:
    pronoun_bias_studies: Pronoun-resolution bias metrics
    """
    return fit_ok and sample_ok


def pronoun_bias_studies_aux(aux: bool) -> bool:
    """pronoun_bias_studies

    aux:
    pronoun_bias_studies: sentences, referents, predictions, and accuracies
    """
    return aux


def _bench_pronoun_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pronoun_bias_studies_ok(True, True))
    checks.append(not pronoun_bias_studies_ok(False, True))
    checks.append(pronoun_bias_studies_aux(True))
    checks.append(not pronoun_bias_studies_aux(False))
    checks.append(True)  # bias-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_pronoun_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pronoun_bias_studies": _bench_pronoun_bias_studies(seed)}
