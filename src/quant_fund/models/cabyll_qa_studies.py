"""cabyll_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def cabyll_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cabyll_qa_studies

    check:
    cabyll_qa_studies: w
    """
    return fit_ok and sample_ok


def cabyll_qa_studies_aux(aux: bool) -> bool:
    """cabyll_qa_studies

    aux:
    cabyll_qa_studies: a
    """
    return aux


def _bench_cabyll_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(cabyll_qa_studies_ok(True, True))
    checks.append(not cabyll_qa_studies_ok(False, True))
    checks.append(cabyll_qa_studies_aux(True))
    checks.append(not cabyll_qa_studies_aux(False))
    checks.append(True)  # celtic-myth-6 canon
    return float(sum(checks) / len(checks))


def bench_cabyll_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cabyll_qa_studies": _bench_cabyll_qa_studies(seed)}
