"""fennel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def fennel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fennel_qa_studies

    check:
    fennel_qa_studies: FennelQA metrics
    """
    return fit_ok and sample_ok


def fennel_qa_studies_aux(aux: bool) -> bool:
    """fennel_qa_studies

    aux:
    fennel_qa_studies: fennels, bulbs, answers, and scores
    """
    return aux


def _bench_fennel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fennel_qa_studies_ok(True, True))
    checks.append(not fennel_qa_studies_ok(False, True))
    checks.append(fennel_qa_studies_aux(True))
    checks.append(not fennel_qa_studies_aux(False))
    checks.append(True)  # spice canon
    return float(sum(checks) / len(checks))


def bench_fennel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fennel_qa_studies": _bench_fennel_qa_studies(seed)}
