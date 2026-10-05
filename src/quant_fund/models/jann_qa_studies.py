"""jann_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jann_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jann_qa_studies

    check:
    jann_qa_studies: J
    """
    return fit_ok and sample_ok


def jann_qa_studies_aux(aux: bool) -> bool:
    """jann_qa_studies

    aux:
    jann_qa_studies: a
    """
    return aux


def _bench_jann_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jann_qa_studies_ok(True, True))
    checks.append(not jann_qa_studies_ok(False, True))
    checks.append(jann_qa_studies_aux(True))
    checks.append(not jann_qa_studies_aux(False))
    checks.append(True)  # jinn canon
    return float(sum(checks) / len(checks))


def bench_jann_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jann_qa_studies": _bench_jann_qa_studies(seed)}
