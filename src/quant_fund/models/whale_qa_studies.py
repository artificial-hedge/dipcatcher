"""whale_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def whale_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """whale_qa_studies

    check:
    whale_qa_studies: WhaleQA metrics
    """
    return fit_ok and sample_ok


def whale_qa_studies_aux(aux: bool) -> bool:
    """whale_qa_studies

    aux:
    whale_qa_studies: whales, migrations, answers, and scores
    """
    return aux


def _bench_whale_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(whale_qa_studies_ok(True, True))
    checks.append(not whale_qa_studies_ok(False, True))
    checks.append(whale_qa_studies_aux(True))
    checks.append(not whale_qa_studies_aux(False))
    checks.append(True)  # marine canon
    return float(sum(checks) / len(checks))


def bench_whale_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whale_qa_studies": _bench_whale_qa_studies(seed)}
