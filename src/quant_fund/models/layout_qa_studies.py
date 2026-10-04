"""layout_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def layout_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """layout_qa_studies

    check:
    layout_qa_studies: LayoutQA metrics
    """
    return fit_ok and sample_ok


def layout_qa_studies_aux(aux: bool) -> bool:
    """layout_qa_studies

    aux:
    layout_qa_studies: pages, elements, answers, and scores
    """
    return aux


def _bench_layout_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(layout_qa_studies_ok(True, True))
    checks.append(not layout_qa_studies_ok(False, True))
    checks.append(layout_qa_studies_aux(True))
    checks.append(not layout_qa_studies_aux(False))
    checks.append(True)  # design-spec canon
    return float(sum(checks) / len(checks))


def bench_layout_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_layout_qa_studies": _bench_layout_qa_studies(seed)}
