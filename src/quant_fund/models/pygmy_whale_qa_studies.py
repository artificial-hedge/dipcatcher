"""pygmy_whale_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def pygmy_whale_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pygmy_whale_qa_studies

    check:
    pygmy_whale_qa_studies: PygmyWhaleQA metrics
    """
    return fit_ok and sample_ok


def pygmy_whale_qa_studies_aux(aux: bool) -> bool:
    """pygmy_whale_qa_studies

    aux:
    pygmy_whale_qa_studies: pygmy whales, deep trenches, answers, and scores
    """
    return aux


def _bench_pygmy_whale_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(pygmy_whale_qa_studies_ok(True, True))
    checks.append(not pygmy_whale_qa_studies_ok(False, True))
    checks.append(pygmy_whale_qa_studies_aux(True))
    checks.append(not pygmy_whale_qa_studies_aux(False))
    checks.append(True)  # ocean-mammal canon
    return float(sum(checks) / len(checks))


def bench_pygmy_whale_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pygmy_whale_qa_studies": _bench_pygmy_whale_qa_studies(seed)}
