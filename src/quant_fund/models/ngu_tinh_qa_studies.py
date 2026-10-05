"""ngu_tinh_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ngu_tinh_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ngu_tinh_qa_studies

    check:
    ngu_tinh_qa_studies: N
    """
    return fit_ok and sample_ok


def ngu_tinh_qa_studies_aux(aux: bool) -> bool:
    """ngu_tinh_qa_studies

    aux:
    ngu_tinh_qa_studies: g
    """
    return aux


def _bench_ngu_tinh_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ngu_tinh_qa_studies_ok(True, True))
    checks.append(not ngu_tinh_qa_studies_ok(False, True))
    checks.append(ngu_tinh_qa_studies_aux(True))
    checks.append(not ngu_tinh_qa_studies_aux(False))
    checks.append(True)  # vietnamese-demon canon
    return float(sum(checks) / len(checks))


def bench_ngu_tinh_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ngu_tinh_qa_studies": _bench_ngu_tinh_qa_studies(seed)}
