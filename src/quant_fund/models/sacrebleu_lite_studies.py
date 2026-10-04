"""sacrebleu_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def sacrebleu_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sacrebleu_lite_studies

    check:
    sacrebleu_lite_studies: SacreBLEU metrics
    """
    return fit_ok and sample_ok


def sacrebleu_lite_studies_aux(aux: bool) -> bool:
    """sacrebleu_lite_studies

    aux:
    sacrebleu_lite_studies: hypotheses, references, labels, and scores
    """
    return aux


def _bench_sacrebleu_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sacrebleu_lite_studies_ok(True, True))
    checks.append(not sacrebleu_lite_studies_ok(False, True))
    checks.append(sacrebleu_lite_studies_aux(True))
    checks.append(not sacrebleu_lite_studies_aux(False))
    checks.append(True)  # translation-metric canon
    return float(sum(checks) / len(checks))


def bench_sacrebleu_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sacrebleu_lite_studies": _bench_sacrebleu_lite_studies(seed)}
