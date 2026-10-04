"""gannet_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gannet_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gannet_qa_studies

    check:
    gannet_qa_studies: GannetQA metrics
    """
    return fit_ok and sample_ok


def gannet_qa_studies_aux(aux: bool) -> bool:
    """gannet_qa_studies

    aux:
    gannet_qa_studies: gannets, plunges, answers, and scores
    """
    return aux


def _bench_gannet_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gannet_qa_studies_ok(True, True))
    checks.append(not gannet_qa_studies_ok(False, True))
    checks.append(gannet_qa_studies_aux(True))
    checks.append(not gannet_qa_studies_aux(False))
    checks.append(True)  # seabird canon
    return float(sum(checks) / len(checks))


def bench_gannet_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gannet_qa_studies": _bench_gannet_qa_studies(seed)}
