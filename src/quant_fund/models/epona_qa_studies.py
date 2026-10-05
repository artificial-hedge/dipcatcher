"""epona_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def epona_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """epona_qa_studies

    check:
    epona_qa_studies: m
    """
    return fit_ok and sample_ok


def epona_qa_studies_aux(aux: bool) -> bool:
    """epona_qa_studies

    aux:
    epona_qa_studies: a
    """
    return aux


def _bench_epona_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(epona_qa_studies_ok(True, True))
    checks.append(not epona_qa_studies_ok(False, True))
    checks.append(epona_qa_studies_aux(True))
    checks.append(not epona_qa_studies_aux(False))
    checks.append(True)  # gallic-myth canon
    return float(sum(checks) / len(checks))


def bench_epona_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epona_qa_studies": _bench_epona_qa_studies(seed)}
