"""tlaloc2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tlaloc2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tlaloc2_qa_studies

    check:
    tlaloc2_qa_studies: Tlaloc2QA metrics
    """
    return fit_ok and sample_ok


def tlaloc2_qa_studies_aux(aux: bool) -> bool:
    """tlaloc2_qa_studies

    aux:
    tlaloc2_qa_studies: tlaloc2, rain givers, answers, and scores
    """
    return aux


def _bench_tlaloc2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tlaloc2_qa_studies_ok(True, True))
    checks.append(not tlaloc2_qa_studies_ok(False, True))
    checks.append(tlaloc2_qa_studies_aux(True))
    checks.append(not tlaloc2_qa_studies_aux(False))
    checks.append(True)  # aztec-deity-5 canon
    return float(sum(checks) / len(checks))


def bench_tlaloc2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tlaloc2_qa_studies": _bench_tlaloc2_qa_studies(seed)}
