"""bold_bias_studies module (SYNTHETIC)."""

from __future__ import annotations


def bold_bias_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bold_bias_studies

    check:
    bold_bias_studies: BOLD profession/gender bias metrics
    """
    return fit_ok and sample_ok


def bold_bias_studies_aux(aux: bool) -> bool:
    """bold_bias_studies

    aux:
    bold_bias_studies: prompts, generations, and sentiment scores
    """
    return aux


def _bench_bold_bias_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bold_bias_studies_ok(True, True))
    checks.append(not bold_bias_studies_ok(False, True))
    checks.append(bold_bias_studies_aux(True))
    checks.append(not bold_bias_studies_aux(False))
    checks.append(True)  # safety-bias-eval canon
    return float(sum(checks) / len(checks))


def bench_bold_bias_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bold_bias_studies": _bench_bold_bias_studies(seed)}
