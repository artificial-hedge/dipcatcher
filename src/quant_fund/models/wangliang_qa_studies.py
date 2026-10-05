"""wangliang_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def wangliang_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wangliang_qa_studies

    check:
    wangliang_qa_studies: W
    """
    return fit_ok and sample_ok


def wangliang_qa_studies_aux(aux: bool) -> bool:
    """wangliang_qa_studies

    aux:
    wangliang_qa_studies: a
    """
    return aux


def _bench_wangliang_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wangliang_qa_studies_ok(True, True))
    checks.append(not wangliang_qa_studies_ok(False, True))
    checks.append(wangliang_qa_studies_aux(True))
    checks.append(not wangliang_qa_studies_aux(False))
    checks.append(True)  # chinese-underworld canon
    return float(sum(checks) / len(checks))


def bench_wangliang_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wangliang_qa_studies": _bench_wangliang_qa_studies(seed)}
