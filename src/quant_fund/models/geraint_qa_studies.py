"""geraint_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def geraint_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geraint_qa_studies

    check:
    geraint_qa_studies: d
    """
    return fit_ok and sample_ok


def geraint_qa_studies_aux(aux: bool) -> bool:
    """geraint_qa_studies

    aux:
    geraint_qa_studies: e
    """
    return aux


def _bench_geraint_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(geraint_qa_studies_ok(True, True))
    checks.append(not geraint_qa_studies_ok(False, True))
    checks.append(geraint_qa_studies_aux(True))
    checks.append(not geraint_qa_studies_aux(False))
    checks.append(True)  # arthurian-6 canon
    return float(sum(checks) / len(checks))


def bench_geraint_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geraint_qa_studies": _bench_geraint_qa_studies(seed)}
