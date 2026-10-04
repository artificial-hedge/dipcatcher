"""multirc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def multirc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """multirc_lite_studies

    check:
    multirc_lite_studies: MultiRC metrics
    """
    return fit_ok and sample_ok


def multirc_lite_studies_aux(aux: bool) -> bool:
    """multirc_lite_studies

    aux:
    multirc_lite_studies: passages, questions, answers, and accuracies
    """
    return aux


def _bench_multirc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(multirc_lite_studies_ok(True, True))
    checks.append(not multirc_lite_studies_ok(False, True))
    checks.append(multirc_lite_studies_aux(True))
    checks.append(not multirc_lite_studies_aux(False))
    checks.append(True)  # reading-comp-3 canon
    return float(sum(checks) / len(checks))


def bench_multirc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multirc_lite_studies": _bench_multirc_lite_studies(seed)}
