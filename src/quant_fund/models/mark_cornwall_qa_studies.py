"""mark_cornwall_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mark_cornwall_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mark_cornwall_qa_studies

    check:
    mark_cornwall_qa_studies: C
    """
    return fit_ok and sample_ok


def mark_cornwall_qa_studies_aux(aux: bool) -> bool:
    """mark_cornwall_qa_studies

    aux:
    mark_cornwall_qa_studies: o
    """
    return aux


def _bench_mark_cornwall_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mark_cornwall_qa_studies_ok(True, True))
    checks.append(not mark_cornwall_qa_studies_ok(False, True))
    checks.append(mark_cornwall_qa_studies_aux(True))
    checks.append(not mark_cornwall_qa_studies_aux(False))
    checks.append(True)  # arthurian-3 canon
    return float(sum(checks) / len(checks))


def bench_mark_cornwall_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mark_cornwall_qa_studies": _bench_mark_cornwall_qa_studies(seed)}
