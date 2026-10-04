"""hare_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def hare_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hare_qa_studies

    check:
    hare_qa_studies: HareQA metrics
    """
    return fit_ok and sample_ok


def hare_qa_studies_aux(aux: bool) -> bool:
    """hare_qa_studies

    aux:
    hare_qa_studies: hares, open fellside, answers, and scores
    """
    return aux


def _bench_hare_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(hare_qa_studies_ok(True, True))
    checks.append(not hare_qa_studies_ok(False, True))
    checks.append(hare_qa_studies_aux(True))
    checks.append(not hare_qa_studies_aux(False))
    checks.append(True)  # small-mammal canon
    return float(sum(checks) / len(checks))


def bench_hare_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hare_qa_studies": _bench_hare_qa_studies(seed)}
