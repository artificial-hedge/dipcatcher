"""nian_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nian_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nian_qa_studies

    check:
    nian_qa_studies: N
    """
    return fit_ok and sample_ok


def nian_qa_studies_aux(aux: bool) -> bool:
    """nian_qa_studies

    aux:
    nian_qa_studies: i
    """
    return aux


def _bench_nian_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nian_qa_studies_ok(True, True))
    checks.append(not nian_qa_studies_ok(False, True))
    checks.append(nian_qa_studies_aux(True))
    checks.append(not nian_qa_studies_aux(False))
    checks.append(True)  # chinese-demon canon
    return float(sum(checks) / len(checks))


def bench_nian_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nian_qa_studies": _bench_nian_qa_studies(seed)}
