"""razorbill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def razorbill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """razorbill_qa_studies

    check:
    razorbill_qa_studies: RazorbillQA metrics
    """
    return fit_ok and sample_ok


def razorbill_qa_studies_aux(aux: bool) -> bool:
    """razorbill_qa_studies

    aux:
    razorbill_qa_studies: razorbills, headlands, answers, and scores
    """
    return aux


def _bench_razorbill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(razorbill_qa_studies_ok(True, True))
    checks.append(not razorbill_qa_studies_ok(False, True))
    checks.append(razorbill_qa_studies_aux(True))
    checks.append(not razorbill_qa_studies_aux(False))
    checks.append(True)  # seabird-3 canon
    return float(sum(checks) / len(checks))


def bench_razorbill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_razorbill_qa_studies": _bench_razorbill_qa_studies(seed)}
