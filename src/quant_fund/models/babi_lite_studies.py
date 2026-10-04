"""babi_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def babi_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """babi_lite_studies

    check:
    babi_lite_studies: bAbI task metrics
    """
    return fit_ok and sample_ok


def babi_lite_studies_aux(aux: bool) -> bool:
    """babi_lite_studies

    aux:
    babi_lite_studies: stories, questions, answers, and accuracies
    """
    return aux


def _bench_babi_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(babi_lite_studies_ok(True, True))
    checks.append(not babi_lite_studies_ok(False, True))
    checks.append(babi_lite_studies_aux(True))
    checks.append(not babi_lite_studies_aux(False))
    checks.append(True)  # QA-exotics canon
    return float(sum(checks) / len(checks))


def bench_babi_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_babi_lite_studies": _bench_babi_lite_studies(seed)}
