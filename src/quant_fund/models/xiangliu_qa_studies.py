"""xiangliu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def xiangliu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """xiangliu_qa_studies

    check:
    xiangliu_qa_studies: X
    """
    return fit_ok and sample_ok


def xiangliu_qa_studies_aux(aux: bool) -> bool:
    """xiangliu_qa_studies

    aux:
    xiangliu_qa_studies: i
    """
    return aux


def _bench_xiangliu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(xiangliu_qa_studies_ok(True, True))
    checks.append(not xiangliu_qa_studies_ok(False, True))
    checks.append(xiangliu_qa_studies_aux(True))
    checks.append(not xiangliu_qa_studies_aux(False))
    checks.append(True)  # chinese-demon canon
    return float(sum(checks) / len(checks))


def bench_xiangliu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_xiangliu_qa_studies": _bench_xiangliu_qa_studies(seed)}
