"""knot_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def knot_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """knot_qa_studies

    check:
    knot_qa_studies: KnotQA metrics
    """
    return fit_ok and sample_ok


def knot_qa_studies_aux(aux: bool) -> bool:
    """knot_qa_studies

    aux:
    knot_qa_studies: knots, estuaries, answers, and scores
    """
    return aux


def _bench_knot_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(knot_qa_studies_ok(True, True))
    checks.append(not knot_qa_studies_ok(False, True))
    checks.append(knot_qa_studies_aux(True))
    checks.append(not knot_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_knot_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_qa_studies": _bench_knot_qa_studies(seed)}
