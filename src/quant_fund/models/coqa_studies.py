"""coqa_studies module (SYNTHETIC)."""

from __future__ import annotations


def coqa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """coqa_studies

    check:
    coqa_studies: CoQA conversational QA and turn-level F1
    """
    return fit_ok and sample_ok


def coqa_studies_aux(aux: bool) -> bool:
    """coqa_studies

    aux:
    coqa_studies: dialogue turns, rationales, and coherence
    """
    return aux


def _bench_coqa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(coqa_studies_ok(True, True))
    checks.append(not coqa_studies_ok(False, True))
    checks.append(coqa_studies_aux(True))
    checks.append(not coqa_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_coqa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coqa_studies": _bench_coqa_studies(seed)}
