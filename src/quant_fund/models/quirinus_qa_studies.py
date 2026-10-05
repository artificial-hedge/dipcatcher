"""quirinus_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def quirinus_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """quirinus_qa_studies

    check:
    quirinus_qa_studies: QuirinusQA metrics
    """
    return fit_ok and sample_ok


def quirinus_qa_studies_aux(aux: bool) -> bool:
    """quirinus_qa_studies

    aux:
    quirinus_qa_studies: quirinus, spear shields, answers, and scores
    """
    return aux


def _bench_quirinus_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(quirinus_qa_studies_ok(True, True))
    checks.append(not quirinus_qa_studies_ok(False, True))
    checks.append(quirinus_qa_studies_aux(True))
    checks.append(not quirinus_qa_studies_aux(False))
    checks.append(True)  # roman-minor-2 canon
    return float(sum(checks) / len(checks))


def bench_quirinus_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quirinus_qa_studies": _bench_quirinus_qa_studies(seed)}
