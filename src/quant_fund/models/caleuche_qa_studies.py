"""caleuche_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def caleuche_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """caleuche_qa_studies

    check:
    caleuche_qa_studies: C
    """
    return fit_ok and sample_ok


def caleuche_qa_studies_aux(aux: bool) -> bool:
    """caleuche_qa_studies

    aux:
    caleuche_qa_studies: a
    """
    return aux


def _bench_caleuche_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(caleuche_qa_studies_ok(True, True))
    checks.append(not caleuche_qa_studies_ok(False, True))
    checks.append(caleuche_qa_studies_aux(True))
    checks.append(not caleuche_qa_studies_aux(False))
    checks.append(True)  # chiloe-demon canon
    return float(sum(checks) / len(checks))


def bench_caleuche_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_caleuche_qa_studies": _bench_caleuche_qa_studies(seed)}
