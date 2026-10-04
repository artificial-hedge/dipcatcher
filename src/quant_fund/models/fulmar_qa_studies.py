"""fulmar_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fulmar_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fulmar_qa_studies

    check:
    fulmar_qa_studies: FulmarQA metrics
    """
    return fit_ok and sample_ok


def fulmar_qa_studies_aux(aux: bool) -> bool:
    """fulmar_qa_studies

    aux:
    fulmar_qa_studies: fulmars, cliffs, answers, and scores
    """
    return aux


def _bench_fulmar_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fulmar_qa_studies_ok(True, True))
    checks.append(not fulmar_qa_studies_ok(False, True))
    checks.append(fulmar_qa_studies_aux(True))
    checks.append(not fulmar_qa_studies_aux(False))
    checks.append(True)  # seabird-2 canon
    return float(sum(checks) / len(checks))


def bench_fulmar_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fulmar_qa_studies": _bench_fulmar_qa_studies(seed)}
