"""dream_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def dream_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dream_lite_studies

    check:
    dream_lite_studies: DREAM commonsense metrics
    """
    return fit_ok and sample_ok


def dream_lite_studies_aux(aux: bool) -> bool:
    """dream_lite_studies

    aux:
    dream_lite_studies: dialogs, questions, answers, and scores
    """
    return aux


def _bench_dream_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dream_lite_studies_ok(True, True))
    checks.append(not dream_lite_studies_ok(False, True))
    checks.append(dream_lite_studies_aux(True))
    checks.append(not dream_lite_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_dream_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dream_lite_studies": _bench_dream_lite_studies(seed)}
