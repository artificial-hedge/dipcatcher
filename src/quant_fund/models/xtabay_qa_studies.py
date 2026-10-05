"""xtabay_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def xtabay_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """xtabay_qa_studies

    check:
    xtabay_qa_studies: X
    """
    return fit_ok and sample_ok


def xtabay_qa_studies_aux(aux: bool) -> bool:
    """xtabay_qa_studies

    aux:
    xtabay_qa_studies: t
    """
    return aux


def _bench_xtabay_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(xtabay_qa_studies_ok(True, True))
    checks.append(not xtabay_qa_studies_ok(False, True))
    checks.append(xtabay_qa_studies_aux(True))
    checks.append(not xtabay_qa_studies_aux(False))
    checks.append(True)  # mesoamerican-demon canon
    return float(sum(checks) / len(checks))


def bench_xtabay_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_xtabay_qa_studies": _bench_xtabay_qa_studies(seed)}
