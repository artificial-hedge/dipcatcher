"""nilgiri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def nilgiri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nilgiri_qa_studies

    check:
    nilgiri_qa_studies: NilgiriQA metrics
    """
    return fit_ok and sample_ok


def nilgiri_qa_studies_aux(aux: bool) -> bool:
    """nilgiri_qa_studies

    aux:
    nilgiri_qa_studies: nilgiri tahrs, montane grasslands, answers, and scores
    """
    return aux


def _bench_nilgiri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nilgiri_qa_studies_ok(True, True))
    checks.append(not nilgiri_qa_studies_ok(False, True))
    checks.append(nilgiri_qa_studies_aux(True))
    checks.append(not nilgiri_qa_studies_aux(False))
    checks.append(True)  # alpine-ridgeline canon
    return float(sum(checks) / len(checks))


def bench_nilgiri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nilgiri_qa_studies": _bench_nilgiri_qa_studies(seed)}
