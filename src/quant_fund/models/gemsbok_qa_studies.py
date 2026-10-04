"""gemsbok_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gemsbok_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gemsbok_qa_studies

    check:
    gemsbok_qa_studies: GemsbokQA metrics
    """
    return fit_ok and sample_ok


def gemsbok_qa_studies_aux(aux: bool) -> bool:
    """gemsbok_qa_studies

    aux:
    gemsbok_qa_studies: gemsboks, kalahari pans, answers, and scores
    """
    return aux


def _bench_gemsbok_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gemsbok_qa_studies_ok(True, True))
    checks.append(not gemsbok_qa_studies_ok(False, True))
    checks.append(gemsbok_qa_studies_aux(True))
    checks.append(not gemsbok_qa_studies_aux(False))
    checks.append(True)  # plains-game canon
    return float(sum(checks) / len(checks))


def bench_gemsbok_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gemsbok_qa_studies": _bench_gemsbok_qa_studies(seed)}
