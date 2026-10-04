"""oystercatcher_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def oystercatcher_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """oystercatcher_qa_studies

    check:
    oystercatcher_qa_studies: OystercatcherQA metrics
    """
    return fit_ok and sample_ok


def oystercatcher_qa_studies_aux(aux: bool) -> bool:
    """oystercatcher_qa_studies

    aux:
    oystercatcher_qa_studies: oystercatchers, shores, answers, and scores
    """
    return aux


def _bench_oystercatcher_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(oystercatcher_qa_studies_ok(True, True))
    checks.append(not oystercatcher_qa_studies_ok(False, True))
    checks.append(oystercatcher_qa_studies_aux(True))
    checks.append(not oystercatcher_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_oystercatcher_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oystercatcher_qa_studies": _bench_oystercatcher_qa_studies(seed)}
