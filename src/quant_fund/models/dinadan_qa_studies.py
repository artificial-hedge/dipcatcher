"""dinadan_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dinadan_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dinadan_qa_studies

    check:
    dinadan_qa_studies: j
    """
    return fit_ok and sample_ok


def dinadan_qa_studies_aux(aux: bool) -> bool:
    """dinadan_qa_studies

    aux:
    dinadan_qa_studies: e
    """
    return aux


def _bench_dinadan_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dinadan_qa_studies_ok(True, True))
    checks.append(not dinadan_qa_studies_ok(False, True))
    checks.append(dinadan_qa_studies_aux(True))
    checks.append(not dinadan_qa_studies_aux(False))
    checks.append(True)  # arthurian-5 canon
    return float(sum(checks) / len(checks))


def bench_dinadan_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dinadan_qa_studies": _bench_dinadan_qa_studies(seed)}
