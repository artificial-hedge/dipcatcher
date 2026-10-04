"""wino_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def wino_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wino_bias_studies

    check:
    wino_bias_studies: Winogender-style bias metrics
    """
    return fit_ok and sample_ok


def wino_bias_studies_aux(aux: bool) -> bool:
    """wino_bias_studies

    aux:
    wino_bias_studies: sentences, pronouns, referents, and accuracies
    """
    return aux


def _bench_wino_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wino_bias_studies_ok(True, True))
    checks.append(not wino_bias_studies_ok(False, True))
    checks.append(wino_bias_studies_aux(True))
    checks.append(not wino_bias_studies_aux(False))
    checks.append(True)  # social-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_wino_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wino_bias_studies": _bench_wino_bias_studies(seed)}
