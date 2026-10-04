"""drum_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def drum_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """drum_qa_studies

    check:
    drum_qa_studies: DrumQA metrics
    """
    return fit_ok and sample_ok


def drum_qa_studies_aux(aux: bool) -> bool:
    """drum_qa_studies

    aux:
    drum_qa_studies: drums, beats, answers, and scores
    """
    return aux


def _bench_drum_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(drum_qa_studies_ok(True, True))
    checks.append(not drum_qa_studies_ok(False, True))
    checks.append(drum_qa_studies_aux(True))
    checks.append(not drum_qa_studies_aux(False))
    checks.append(True)  # instrument canon
    return float(sum(checks) / len(checks))


def bench_drum_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drum_qa_studies": _bench_drum_qa_studies(seed)}
