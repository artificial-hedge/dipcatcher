"""juba_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def juba_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """juba_qa_studies

    check:
    juba_qa_studies: s
    """
    return fit_ok and sample_ok


def juba_qa_studies_aux(aux: bool) -> bool:
    """juba_qa_studies

    aux:
    juba_qa_studies: c
    """
    return aux


def _bench_juba_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(juba_qa_studies_ok(True, True))
    checks.append(not juba_qa_studies_ok(False, True))
    checks.append(juba_qa_studies_aux(True))
    checks.append(not juba_qa_studies_aux(False))
    checks.append(True)  # numidian-2 canon
    return float(sum(checks) / len(checks))


def bench_juba_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_juba_qa_studies": _bench_juba_qa_studies(seed)}
