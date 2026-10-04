"""bison_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bison_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bison_qa_studies

    check:
    bison_qa_studies: BisonQA metrics
    """
    return fit_ok and sample_ok


def bison_qa_studies_aux(aux: bool) -> bool:
    """bison_qa_studies

    aux:
    bison_qa_studies: bisons, herds, answers, and scores
    """
    return aux


def _bench_bison_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bison_qa_studies_ok(True, True))
    checks.append(not bison_qa_studies_ok(False, True))
    checks.append(bison_qa_studies_aux(True))
    checks.append(not bison_qa_studies_aux(False))
    checks.append(True)  # forest-mammal canon
    return float(sum(checks) / len(checks))


def bench_bison_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bison_qa_studies": _bench_bison_qa_studies(seed)}
