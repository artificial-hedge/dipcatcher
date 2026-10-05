"""cybele2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cybele2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cybele2_qa_studies

    check:
    cybele2_qa_studies: Cybele2QA metrics
    """
    return fit_ok and sample_ok


def cybele2_qa_studies_aux(aux: bool) -> bool:
    """cybele2_qa_studies

    aux:
    cybele2_qa_studies: cybele2, lion mothers, answers, and scores
    """
    return aux


def _bench_cybele2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cybele2_qa_studies_ok(True, True))
    checks.append(not cybele2_qa_studies_ok(False, True))
    checks.append(cybele2_qa_studies_aux(True))
    checks.append(not cybele2_qa_studies_aux(False))
    checks.append(True)  # phrygian-myth canon
    return float(sum(checks) / len(checks))


def bench_cybele2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cybele2_qa_studies": _bench_cybele2_qa_studies(seed)}
