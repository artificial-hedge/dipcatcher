"""sardine_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sardine_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sardine_qa_studies

    check:
    sardine_qa_studies: SardineQA metrics
    """
    return fit_ok and sample_ok


def sardine_qa_studies_aux(aux: bool) -> bool:
    """sardine_qa_studies

    aux:
    sardine_qa_studies: sardines, bait balls, answers, and scores
    """
    return aux


def _bench_sardine_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sardine_qa_studies_ok(True, True))
    checks.append(not sardine_qa_studies_ok(False, True))
    checks.append(sardine_qa_studies_aux(True))
    checks.append(not sardine_qa_studies_aux(False))
    checks.append(True)  # pelagic-fish canon
    return float(sum(checks) / len(checks))


def bench_sardine_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sardine_qa_studies": _bench_sardine_qa_studies(seed)}
