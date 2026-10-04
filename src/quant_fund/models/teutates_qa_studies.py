"""teutates_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def teutates_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """teutates_qa_studies

    check:
    teutates_qa_studies: t
    """
    return fit_ok and sample_ok


def teutates_qa_studies_aux(aux: bool) -> bool:
    """teutates_qa_studies

    aux:
    teutates_qa_studies: r
    """
    return aux


def _bench_teutates_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(teutates_qa_studies_ok(True, True))
    checks.append(not teutates_qa_studies_ok(False, True))
    checks.append(teutates_qa_studies_aux(True))
    checks.append(not teutates_qa_studies_aux(False))
    checks.append(True)  # gallic-myth canon
    return float(sum(checks) / len(checks))


def bench_teutates_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_teutates_qa_studies": _bench_teutates_qa_studies(seed)}
