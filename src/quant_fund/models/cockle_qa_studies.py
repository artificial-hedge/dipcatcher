"""cockle_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cockle_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cockle_qa_studies

    check:
    cockle_qa_studies: CockleQA metrics
    """
    return fit_ok and sample_ok


def cockle_qa_studies_aux(aux: bool) -> bool:
    """cockle_qa_studies

    aux:
    cockle_qa_studies: cockles, tidal flats, answers, and scores
    """
    return aux


def _bench_cockle_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cockle_qa_studies_ok(True, True))
    checks.append(not cockle_qa_studies_ok(False, True))
    checks.append(cockle_qa_studies_aux(True))
    checks.append(not cockle_qa_studies_aux(False))
    checks.append(True)  # mollusk canon
    return float(sum(checks) / len(checks))


def bench_cockle_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cockle_qa_studies": _bench_cockle_qa_studies(seed)}
