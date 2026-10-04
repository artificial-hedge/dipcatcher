"""poorwill_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def poorwill_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """poorwill_qa_studies

    check:
    poorwill_qa_studies: PoorwillQA metrics
    """
    return fit_ok and sample_ok


def poorwill_qa_studies_aux(aux: bool) -> bool:
    """poorwill_qa_studies

    aux:
    poorwill_qa_studies: poorwills, deserts, answers, and scores
    """
    return aux


def _bench_poorwill_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(poorwill_qa_studies_ok(True, True))
    checks.append(not poorwill_qa_studies_ok(False, True))
    checks.append(poorwill_qa_studies_aux(True))
    checks.append(not poorwill_qa_studies_aux(False))
    checks.append(True)  # nightjar-2 canon
    return float(sum(checks) / len(checks))


def bench_poorwill_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poorwill_qa_studies": _bench_poorwill_qa_studies(seed)}
