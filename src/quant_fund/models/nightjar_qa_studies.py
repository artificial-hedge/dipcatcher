"""nightjar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nightjar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nightjar_qa_studies

    check:
    nightjar_qa_studies: NightjarQA metrics
    """
    return fit_ok and sample_ok


def nightjar_qa_studies_aux(aux: bool) -> bool:
    """nightjar_qa_studies

    aux:
    nightjar_qa_studies: nightjars, clearings, answers, and scores
    """
    return aux


def _bench_nightjar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nightjar_qa_studies_ok(True, True))
    checks.append(not nightjar_qa_studies_ok(False, True))
    checks.append(nightjar_qa_studies_aux(True))
    checks.append(not nightjar_qa_studies_aux(False))
    checks.append(True)  # nightbird canon
    return float(sum(checks) / len(checks))


def bench_nightjar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nightjar_qa_studies": _bench_nightjar_qa_studies(seed)}
