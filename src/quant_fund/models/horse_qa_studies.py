"""horse_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def horse_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """horse_qa_studies

    check:
    horse_qa_studies: HorseQA metrics
    """
    return fit_ok and sample_ok


def horse_qa_studies_aux(aux: bool) -> bool:
    """horse_qa_studies

    aux:
    horse_qa_studies: horses, saddles, answers, and scores
    """
    return aux


def _bench_horse_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(horse_qa_studies_ok(True, True))
    checks.append(not horse_qa_studies_ok(False, True))
    checks.append(horse_qa_studies_aux(True))
    checks.append(not horse_qa_studies_aux(False))
    checks.append(True)  # farm canon
    return float(sum(checks) / len(checks))


def bench_horse_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_horse_qa_studies": _bench_horse_qa_studies(seed)}
