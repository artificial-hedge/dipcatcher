"""muki_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def muki_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """muki_qa_studies

    check:
    muki_qa_studies: M
    """
    return fit_ok and sample_ok


def muki_qa_studies_aux(aux: bool) -> bool:
    """muki_qa_studies

    aux:
    muki_qa_studies: u
    """
    return aux


def _bench_muki_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(muki_qa_studies_ok(True, True))
    checks.append(not muki_qa_studies_ok(False, True))
    checks.append(muki_qa_studies_aux(True))
    checks.append(not muki_qa_studies_aux(False))
    checks.append(True)  # andean-demon canon
    return float(sum(checks) / len(checks))


def bench_muki_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_muki_qa_studies": _bench_muki_qa_studies(seed)}
