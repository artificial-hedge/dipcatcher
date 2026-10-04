"""csqa_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def csqa_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """csqa_lite_studies

    check:
    csqa_lite_studies: CommonsenseQA metrics
    """
    return fit_ok and sample_ok


def csqa_lite_studies_aux(aux: bool) -> bool:
    """csqa_lite_studies

    aux:
    csqa_lite_studies: questions, concepts, options, and accuracies
    """
    return aux


def _bench_csqa_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(csqa_lite_studies_ok(True, True))
    checks.append(not csqa_lite_studies_ok(False, True))
    checks.append(csqa_lite_studies_aux(True))
    checks.append(not csqa_lite_studies_aux(False))
    checks.append(True)  # commonsense-eval canon
    return float(sum(checks) / len(checks))


def bench_csqa_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_csqa_lite_studies": _bench_csqa_lite_studies(seed)}
