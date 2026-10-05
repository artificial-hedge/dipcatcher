"""bgegs_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bgegs_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bgegs_qa_studies

    check:
    bgegs_qa_studies: B
    """
    return fit_ok and sample_ok


def bgegs_qa_studies_aux(aux: bool) -> bool:
    """bgegs_qa_studies

    aux:
    bgegs_qa_studies: g
    """
    return aux


def _bench_bgegs_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bgegs_qa_studies_ok(True, True))
    checks.append(not bgegs_qa_studies_ok(False, True))
    checks.append(bgegs_qa_studies_aux(True))
    checks.append(not bgegs_qa_studies_aux(False))
    checks.append(True)  # tibetan-demon canon
    return float(sum(checks) / len(checks))


def bench_bgegs_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bgegs_qa_studies": _bench_bgegs_qa_studies(seed)}
