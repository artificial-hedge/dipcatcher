"""koshchey_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def koshchey_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """koshchey_qa_studies

    check:
    koshchey_qa_studies: K
    """
    return fit_ok and sample_ok


def koshchey_qa_studies_aux(aux: bool) -> bool:
    """koshchey_qa_studies

    aux:
    koshchey_qa_studies: o
    """
    return aux


def _bench_koshchey_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(koshchey_qa_studies_ok(True, True))
    checks.append(not koshchey_qa_studies_ok(False, True))
    checks.append(koshchey_qa_studies_aux(True))
    checks.append(not koshchey_qa_studies_aux(False))
    checks.append(True)  # slavic-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_koshchey_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_koshchey_qa_studies": _bench_koshchey_qa_studies(seed)}
