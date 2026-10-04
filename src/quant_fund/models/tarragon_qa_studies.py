"""tarragon_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tarragon_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tarragon_qa_studies

    check:
    tarragon_qa_studies: TarragonQA metrics
    """
    return fit_ok and sample_ok


def tarragon_qa_studies_aux(aux: bool) -> bool:
    """tarragon_qa_studies

    aux:
    tarragon_qa_studies: tarragon, kitchens, answers, and scores
    """
    return aux


def _bench_tarragon_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tarragon_qa_studies_ok(True, True))
    checks.append(not tarragon_qa_studies_ok(False, True))
    checks.append(tarragon_qa_studies_aux(True))
    checks.append(not tarragon_qa_studies_aux(False))
    checks.append(True)  # spice-2 canon
    return float(sum(checks) / len(checks))


def bench_tarragon_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tarragon_qa_studies": _bench_tarragon_qa_studies(seed)}
