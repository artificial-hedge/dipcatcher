"""peregrine_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def peregrine_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """peregrine_qa_studies

    check:
    peregrine_qa_studies: PeregrineQA metrics
    """
    return fit_ok and sample_ok


def peregrine_qa_studies_aux(aux: bool) -> bool:
    """peregrine_qa_studies

    aux:
    peregrine_qa_studies: peregrines, cliffs, answers, and scores
    """
    return aux


def _bench_peregrine_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(peregrine_qa_studies_ok(True, True))
    checks.append(not peregrine_qa_studies_ok(False, True))
    checks.append(peregrine_qa_studies_aux(True))
    checks.append(not peregrine_qa_studies_aux(False))
    checks.append(True)  # raptor-2 canon
    return float(sum(checks) / len(checks))


def bench_peregrine_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_peregrine_qa_studies": _bench_peregrine_qa_studies(seed)}
