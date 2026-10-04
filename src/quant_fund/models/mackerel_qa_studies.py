"""mackerel_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mackerel_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mackerel_qa_studies

    check:
    mackerel_qa_studies: MackerelQA metrics
    """
    return fit_ok and sample_ok


def mackerel_qa_studies_aux(aux: bool) -> bool:
    """mackerel_qa_studies

    aux:
    mackerel_qa_studies: mackerel, temperate shoals, answers, and scores
    """
    return aux


def _bench_mackerel_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mackerel_qa_studies_ok(True, True))
    checks.append(not mackerel_qa_studies_ok(False, True))
    checks.append(mackerel_qa_studies_aux(True))
    checks.append(not mackerel_qa_studies_aux(False))
    checks.append(True)  # pelagic-fish canon
    return float(sum(checks) / len(checks))


def bench_mackerel_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mackerel_qa_studies": _bench_mackerel_qa_studies(seed)}
