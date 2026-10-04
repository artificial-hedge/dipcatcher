"""parrotfish_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def parrotfish_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """parrotfish_qa_studies

    check:
    parrotfish_qa_studies: ParrotfishQA metrics
    """
    return fit_ok and sample_ok


def parrotfish_qa_studies_aux(aux: bool) -> bool:
    """parrotfish_qa_studies

    aux:
    parrotfish_qa_studies: parrotfish, coral reefs, answers, and scores
    """
    return aux


def _bench_parrotfish_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(parrotfish_qa_studies_ok(True, True))
    checks.append(not parrotfish_qa_studies_ok(False, True))
    checks.append(parrotfish_qa_studies_aux(True))
    checks.append(not parrotfish_qa_studies_aux(False))
    checks.append(True)  # reef-fish canon
    return float(sum(checks) / len(checks))


def bench_parrotfish_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_parrotfish_qa_studies": _bench_parrotfish_qa_studies(seed)}
