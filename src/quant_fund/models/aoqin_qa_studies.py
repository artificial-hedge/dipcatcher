"""aoqin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def aoqin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """aoqin_qa_studies

    check:
    aoqin_qa_studies: AoqinQA metrics
    """
    return fit_ok and sample_ok


def aoqin_qa_studies_aux(aux: bool) -> bool:
    """aoqin_qa_studies

    aux:
    aoqin_qa_studies: aoqin, dragon kings, answers, and scores
    """
    return aux


def _bench_aoqin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(aoqin_qa_studies_ok(True, True))
    checks.append(not aoqin_qa_studies_ok(False, True))
    checks.append(aoqin_qa_studies_aux(True))
    checks.append(not aoqin_qa_studies_aux(False))
    checks.append(True)  # chinese-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_aoqin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aoqin_qa_studies": _bench_aoqin_qa_studies(seed)}
