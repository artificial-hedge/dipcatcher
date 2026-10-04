"""hoatzin_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hoatzin_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hoatzin_qa_studies

    check:
    hoatzin_qa_studies: HoatzinQA metrics
    """
    return fit_ok and sample_ok


def hoatzin_qa_studies_aux(aux: bool) -> bool:
    """hoatzin_qa_studies

    aux:
    hoatzin_qa_studies: hoatzins, riverine thickets, answers, and scores
    """
    return aux


def _bench_hoatzin_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hoatzin_qa_studies_ok(True, True))
    checks.append(not hoatzin_qa_studies_ok(False, True))
    checks.append(hoatzin_qa_studies_aux(True))
    checks.append(not hoatzin_qa_studies_aux(False))
    checks.append(True)  # cuckoo-turaco canon
    return float(sum(checks) / len(checks))


def bench_hoatzin_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hoatzin_qa_studies": _bench_hoatzin_qa_studies(seed)}
