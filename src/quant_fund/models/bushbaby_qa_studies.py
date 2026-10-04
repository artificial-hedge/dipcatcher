"""bushbaby_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bushbaby_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bushbaby_qa_studies

    check:
    bushbaby_qa_studies: BushbabyQA metrics
    """
    return fit_ok and sample_ok


def bushbaby_qa_studies_aux(aux: bool) -> bool:
    """bushbaby_qa_studies

    aux:
    bushbaby_qa_studies: bushbabies, night thickets, answers, and scores
    """
    return aux


def _bench_bushbaby_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bushbaby_qa_studies_ok(True, True))
    checks.append(not bushbaby_qa_studies_ok(False, True))
    checks.append(bushbaby_qa_studies_aux(True))
    checks.append(not bushbaby_qa_studies_aux(False))
    checks.append(True)  # prosimian canon
    return float(sum(checks) / len(checks))


def bench_bushbaby_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bushbaby_qa_studies": _bench_bushbaby_qa_studies(seed)}
