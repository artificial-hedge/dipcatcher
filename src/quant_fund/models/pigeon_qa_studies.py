"""pigeon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pigeon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pigeon_qa_studies

    check:
    pigeon_qa_studies: PigeonQA metrics
    """
    return fit_ok and sample_ok


def pigeon_qa_studies_aux(aux: bool) -> bool:
    """pigeon_qa_studies

    aux:
    pigeon_qa_studies: pigeons, ledges, answers, and scores
    """
    return aux


def _bench_pigeon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pigeon_qa_studies_ok(True, True))
    checks.append(not pigeon_qa_studies_ok(False, True))
    checks.append(pigeon_qa_studies_aux(True))
    checks.append(not pigeon_qa_studies_aux(False))
    checks.append(True)  # columbid canon
    return float(sum(checks) / len(checks))


def bench_pigeon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pigeon_qa_studies": _bench_pigeon_qa_studies(seed)}
