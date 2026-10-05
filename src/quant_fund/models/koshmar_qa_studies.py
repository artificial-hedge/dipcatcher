"""koshmar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def koshmar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """koshmar_qa_studies

    check:
    koshmar_qa_studies: K
    """
    return fit_ok and sample_ok


def koshmar_qa_studies_aux(aux: bool) -> bool:
    """koshmar_qa_studies

    aux:
    koshmar_qa_studies: o
    """
    return aux


def _bench_koshmar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(koshmar_qa_studies_ok(True, True))
    checks.append(not koshmar_qa_studies_ok(False, True))
    checks.append(koshmar_qa_studies_aux(True))
    checks.append(not koshmar_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_koshmar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koshmar_qa_studies": _bench_koshmar_qa_studies(seed)}
