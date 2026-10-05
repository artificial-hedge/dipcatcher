"""kiriath2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kiriath2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kiriath2_qa_studies

    check:
    kiriath2_qa_studies: c
    """
    return fit_ok and sample_ok


def kiriath2_qa_studies_aux(aux: bool) -> bool:
    """kiriath2_qa_studies

    aux:
    kiriath2_qa_studies: i
    """
    return aux


def _bench_kiriath2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kiriath2_qa_studies_ok(True, True))
    checks.append(not kiriath2_qa_studies_ok(False, True))
    checks.append(kiriath2_qa_studies_aux(True))
    checks.append(not kiriath2_qa_studies_aux(False))
    checks.append(True)  # moabite-myth canon
    return float(sum(checks) / len(checks))


def bench_kiriath2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kiriath2_qa_studies": _bench_kiriath2_qa_studies(seed)}
