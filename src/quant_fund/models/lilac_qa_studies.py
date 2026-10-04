"""lilac_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def lilac_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """lilac_qa_studies

    check:
    lilac_qa_studies: LilacQA metrics
    """
    return fit_ok and sample_ok


def lilac_qa_studies_aux(aux: bool) -> bool:
    """lilac_qa_studies

    aux:
    lilac_qa_studies: lilacs, blossoms, answers, and scores
    """
    return aux


def _bench_lilac_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(lilac_qa_studies_ok(True, True))
    checks.append(not lilac_qa_studies_ok(False, True))
    checks.append(lilac_qa_studies_aux(True))
    checks.append(not lilac_qa_studies_aux(False))
    checks.append(True)  # bloom canon
    return float(sum(checks) / len(checks))


def bench_lilac_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lilac_qa_studies": _bench_lilac_qa_studies(seed)}
