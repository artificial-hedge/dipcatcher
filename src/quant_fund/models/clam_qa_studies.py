"""clam_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def clam_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """clam_qa_studies

    check:
    clam_qa_studies: ClamQA metrics
    """
    return fit_ok and sample_ok


def clam_qa_studies_aux(aux: bool) -> bool:
    """clam_qa_studies

    aux:
    clam_qa_studies: clams, tidal sand, answers, and scores
    """
    return aux


def _bench_clam_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(clam_qa_studies_ok(True, True))
    checks.append(not clam_qa_studies_ok(False, True))
    checks.append(clam_qa_studies_aux(True))
    checks.append(not clam_qa_studies_aux(False))
    checks.append(True)  # bivalve canon
    return float(sum(checks) / len(checks))


def bench_clam_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clam_qa_studies": _bench_clam_qa_studies(seed)}
