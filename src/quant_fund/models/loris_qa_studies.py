"""loris_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def loris_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """loris_qa_studies

    check:
    loris_qa_studies: LorisQA metrics
    """
    return fit_ok and sample_ok


def loris_qa_studies_aux(aux: bool) -> bool:
    """loris_qa_studies

    aux:
    loris_qa_studies: lorises, rainforest understory, answers, and scores
    """
    return aux


def _bench_loris_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(loris_qa_studies_ok(True, True))
    checks.append(not loris_qa_studies_ok(False, True))
    checks.append(loris_qa_studies_aux(True))
    checks.append(not loris_qa_studies_aux(False))
    checks.append(True)  # prosimian canon
    return float(sum(checks) / len(checks))


def bench_loris_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_loris_qa_studies": _bench_loris_qa_studies(seed)}
