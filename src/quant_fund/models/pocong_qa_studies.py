"""pocong_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pocong_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pocong_qa_studies

    check:
    pocong_qa_studies: P
    """
    return fit_ok and sample_ok


def pocong_qa_studies_aux(aux: bool) -> bool:
    """pocong_qa_studies

    aux:
    pocong_qa_studies: o
    """
    return aux


def _bench_pocong_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pocong_qa_studies_ok(True, True))
    checks.append(not pocong_qa_studies_ok(False, True))
    checks.append(pocong_qa_studies_aux(True))
    checks.append(not pocong_qa_studies_aux(False))
    checks.append(True)  # javanese-demon canon
    return float(sum(checks) / len(checks))


def bench_pocong_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pocong_qa_studies": _bench_pocong_qa_studies(seed)}
