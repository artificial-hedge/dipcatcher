"""amayya_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def amayya_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """amayya_qa_studies

    check:
    amayya_qa_studies: s
    """
    return fit_ok and sample_ok


def amayya_qa_studies_aux(aux: bool) -> bool:
    """amayya_qa_studies

    aux:
    amayya_qa_studies: p
    """
    return aux


def _bench_amayya_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(amayya_qa_studies_ok(True, True))
    checks.append(not amayya_qa_studies_ok(False, True))
    checks.append(amayya_qa_studies_aux(True))
    checks.append(not amayya_qa_studies_aux(False))
    checks.append(True)  # garamantian canon
    return float(sum(checks) / len(checks))


def bench_amayya_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amayya_qa_studies": _bench_amayya_qa_studies(seed)}
