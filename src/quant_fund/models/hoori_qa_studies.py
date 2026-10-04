"""hoori_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hoori_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hoori_qa_studies

    check:
    hoori_qa_studies: HooriQA metrics
    """
    return fit_ok and sample_ok


def hoori_qa_studies_aux(aux: bool) -> bool:
    """hoori_qa_studies

    aux:
    hoori_qa_studies: hoori, hook fishers, answers, and scores
    """
    return aux


def _bench_hoori_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hoori_qa_studies_ok(True, True))
    checks.append(not hoori_qa_studies_ok(False, True))
    checks.append(hoori_qa_studies_aux(True))
    checks.append(not hoori_qa_studies_aux(False))
    checks.append(True)  # japanese-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_hoori_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hoori_qa_studies": _bench_hoori_qa_studies(seed)}
