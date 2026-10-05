"""masalai_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def masalai_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """masalai_qa_studies

    check:
    masalai_qa_studies: M
    """
    return fit_ok and sample_ok


def masalai_qa_studies_aux(aux: bool) -> bool:
    """masalai_qa_studies

    aux:
    masalai_qa_studies: a
    """
    return aux


def _bench_masalai_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(masalai_qa_studies_ok(True, True))
    checks.append(not masalai_qa_studies_ok(False, True))
    checks.append(masalai_qa_studies_aux(True))
    checks.append(not masalai_qa_studies_aux(False))
    checks.append(True)  # oceania-demon canon
    return float(sum(checks) / len(checks))


def bench_masalai_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_masalai_qa_studies": _bench_masalai_qa_studies(seed)}
