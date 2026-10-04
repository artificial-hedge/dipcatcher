"""bearded_saki_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bearded_saki_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bearded_saki_qa_studies

    check:
    bearded_saki_qa_studies: BeardedSakiQA metrics
    """
    return fit_ok and sample_ok


def bearded_saki_qa_studies_aux(aux: bool) -> bool:
    """bearded_saki_qa_studies

    aux:
    bearded_saki_qa_studies: bearded sakis, terra firme canopies, answers, and scores
    """
    return aux


def _bench_bearded_saki_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bearded_saki_qa_studies_ok(True, True))
    checks.append(not bearded_saki_qa_studies_ok(False, True))
    checks.append(bearded_saki_qa_studies_aux(True))
    checks.append(not bearded_saki_qa_studies_aux(False))
    checks.append(True)  # primate-4 canon
    return float(sum(checks) / len(checks))


def bench_bearded_saki_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bearded_saki_qa_studies": _bench_bearded_saki_qa_studies(seed)}
