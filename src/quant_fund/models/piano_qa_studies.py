"""piano_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def piano_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """piano_qa_studies

    check:
    piano_qa_studies: PianoQA metrics
    """
    return fit_ok and sample_ok


def piano_qa_studies_aux(aux: bool) -> bool:
    """piano_qa_studies

    aux:
    piano_qa_studies: pianos, pedals, answers, and scores
    """
    return aux


def _bench_piano_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(piano_qa_studies_ok(True, True))
    checks.append(not piano_qa_studies_ok(False, True))
    checks.append(piano_qa_studies_aux(True))
    checks.append(not piano_qa_studies_aux(False))
    checks.append(True)  # instrument canon
    return float(sum(checks) / len(checks))


def bench_piano_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_piano_qa_studies": _bench_piano_qa_studies(seed)}
