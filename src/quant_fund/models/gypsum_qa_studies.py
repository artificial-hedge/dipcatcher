"""gypsum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gypsum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gypsum_qa_studies

    check:
    gypsum_qa_studies: GypsumQA metrics
    """
    return fit_ok and sample_ok


def gypsum_qa_studies_aux(aux: bool) -> bool:
    """gypsum_qa_studies

    aux:
    gypsum_qa_studies: gypsums, playas, answers, and scores
    """
    return aux


def _bench_gypsum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gypsum_qa_studies_ok(True, True))
    checks.append(not gypsum_qa_studies_ok(False, True))
    checks.append(gypsum_qa_studies_aux(True))
    checks.append(not gypsum_qa_studies_aux(False))
    checks.append(True)  # mineral canon
    return float(sum(checks) / len(checks))


def bench_gypsum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gypsum_qa_studies": _bench_gypsum_qa_studies(seed)}
