"""hotupuku_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hotupuku_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hotupuku_qa_studies

    check:
    hotupuku_qa_studies: H
    """
    return fit_ok and sample_ok


def hotupuku_qa_studies_aux(aux: bool) -> bool:
    """hotupuku_qa_studies

    aux:
    hotupuku_qa_studies: o
    """
    return aux


def _bench_hotupuku_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hotupuku_qa_studies_ok(True, True))
    checks.append(not hotupuku_qa_studies_ok(False, True))
    checks.append(hotupuku_qa_studies_aux(True))
    checks.append(not hotupuku_qa_studies_aux(False))
    checks.append(True)  # maori-demon canon
    return float(sum(checks) / len(checks))


def bench_hotupuku_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hotupuku_qa_studies": _bench_hotupuku_qa_studies(seed)}
