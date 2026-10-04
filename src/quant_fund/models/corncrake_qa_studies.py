"""corncrake_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def corncrake_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """corncrake_qa_studies

    check:
    corncrake_qa_studies: CorncrakeQA metrics
    """
    return fit_ok and sample_ok


def corncrake_qa_studies_aux(aux: bool) -> bool:
    """corncrake_qa_studies

    aux:
    corncrake_qa_studies: corncrakes, hay meadows, answers, and scores
    """
    return aux


def _bench_corncrake_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(corncrake_qa_studies_ok(True, True))
    checks.append(not corncrake_qa_studies_ok(False, True))
    checks.append(corncrake_qa_studies_aux(True))
    checks.append(not corncrake_qa_studies_aux(False))
    checks.append(True)  # rail-2 canon
    return float(sum(checks) / len(checks))


def bench_corncrake_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_corncrake_qa_studies": _bench_corncrake_qa_studies(seed)}
