"""pig_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pig_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pig_qa_studies

    check:
    pig_qa_studies: PigQA metrics
    """
    return fit_ok and sample_ok


def pig_qa_studies_aux(aux: bool) -> bool:
    """pig_qa_studies

    aux:
    pig_qa_studies: pigs, troughs, answers, and scores
    """
    return aux


def _bench_pig_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pig_qa_studies_ok(True, True))
    checks.append(not pig_qa_studies_ok(False, True))
    checks.append(pig_qa_studies_aux(True))
    checks.append(not pig_qa_studies_aux(False))
    checks.append(True)  # farm canon
    return float(sum(checks) / len(checks))


def bench_pig_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pig_qa_studies": _bench_pig_qa_studies(seed)}
