"""agaric_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def agaric_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agaric_qa_studies

    check:
    agaric_qa_studies: AgaricQA metrics
    """
    return fit_ok and sample_ok


def agaric_qa_studies_aux(aux: bool) -> bool:
    """agaric_qa_studies

    aux:
    agaric_qa_studies: agarics, meadows, answers, and scores
    """
    return aux


def _bench_agaric_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agaric_qa_studies_ok(True, True))
    checks.append(not agaric_qa_studies_ok(False, True))
    checks.append(agaric_qa_studies_aux(True))
    checks.append(not agaric_qa_studies_aux(False))
    checks.append(True)  # fungi canon
    return float(sum(checks) / len(checks))


def bench_agaric_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agaric_qa_studies": _bench_agaric_qa_studies(seed)}
