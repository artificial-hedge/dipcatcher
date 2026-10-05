"""tiamat2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tiamat2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tiamat2_qa_studies

    check:
    tiamat2_qa_studies: Tiamat2QA metrics
    """
    return fit_ok and sample_ok


def tiamat2_qa_studies_aux(aux: bool) -> bool:
    """tiamat2_qa_studies

    aux:
    tiamat2_qa_studies: tiamat2, salt depths, answers, and scores
    """
    return aux


def _bench_tiamat2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tiamat2_qa_studies_ok(True, True))
    checks.append(not tiamat2_qa_studies_ok(False, True))
    checks.append(tiamat2_qa_studies_aux(True))
    checks.append(not tiamat2_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-3 canon
    return float(sum(checks) / len(checks))


def bench_tiamat2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tiamat2_qa_studies": _bench_tiamat2_qa_studies(seed)}
