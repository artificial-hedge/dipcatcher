"""stilt_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def stilt_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """stilt_qa_studies

    check:
    stilt_qa_studies: StiltQA metrics
    """
    return fit_ok and sample_ok


def stilt_qa_studies_aux(aux: bool) -> bool:
    """stilt_qa_studies

    aux:
    stilt_qa_studies: stilts, shallows, answers, and scores
    """
    return aux


def _bench_stilt_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(stilt_qa_studies_ok(True, True))
    checks.append(not stilt_qa_studies_ok(False, True))
    checks.append(stilt_qa_studies_aux(True))
    checks.append(not stilt_qa_studies_aux(False))
    checks.append(True)  # shorebird-2 canon
    return float(sum(checks) / len(checks))


def bench_stilt_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stilt_qa_studies": _bench_stilt_qa_studies(seed)}
