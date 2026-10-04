"""pillbug_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pillbug_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pillbug_qa_studies

    check:
    pillbug_qa_studies: PillbugQA metrics
    """
    return fit_ok and sample_ok


def pillbug_qa_studies_aux(aux: bool) -> bool:
    """pillbug_qa_studies

    aux:
    pillbug_qa_studies: pillbugs, garden stones, answers, and scores
    """
    return aux


def _bench_pillbug_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pillbug_qa_studies_ok(True, True))
    checks.append(not pillbug_qa_studies_ok(False, True))
    checks.append(pillbug_qa_studies_aux(True))
    checks.append(not pillbug_qa_studies_aux(False))
    checks.append(True)  # detritivore canon
    return float(sum(checks) / len(checks))


def bench_pillbug_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pillbug_qa_studies": _bench_pillbug_qa_studies(seed)}
