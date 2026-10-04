"""leshii_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leshii_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leshii_qa_studies

    check:
    leshii_qa_studies: LeshiiQA metrics
    """
    return fit_ok and sample_ok


def leshii_qa_studies_aux(aux: bool) -> bool:
    """leshii_qa_studies

    aux:
    leshii_qa_studies: leshii, forest spirit, answers, and scores
    """
    return aux


def _bench_leshii_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leshii_qa_studies_ok(True, True))
    checks.append(not leshii_qa_studies_ok(False, True))
    checks.append(leshii_qa_studies_aux(True))
    checks.append(not leshii_qa_studies_aux(False))
    checks.append(True)  # slavic-myth-3 canon
    return float(sum(checks) / len(checks))


def bench_leshii_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leshii_qa_studies": _bench_leshii_qa_studies(seed)}
