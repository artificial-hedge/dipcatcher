"""foliose_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def foliose_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """foliose_qa_studies

    check:
    foliose_qa_studies: FolioseQA metrics
    """
    return fit_ok and sample_ok


def foliose_qa_studies_aux(aux: bool) -> bool:
    """foliose_qa_studies

    aux:
    foliose_qa_studies: folioses, boulders, answers, and scores
    """
    return aux


def _bench_foliose_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(foliose_qa_studies_ok(True, True))
    checks.append(not foliose_qa_studies_ok(False, True))
    checks.append(foliose_qa_studies_aux(True))
    checks.append(not foliose_qa_studies_aux(False))
    checks.append(True)  # lichen canon
    return float(sum(checks) / len(checks))


def bench_foliose_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_foliose_qa_studies": _bench_foliose_qa_studies(seed)}
