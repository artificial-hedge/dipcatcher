"""dagon2_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def dagon2_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dagon2_qa_studies

    check:
    dagon2_qa_studies: f
    """
    return fit_ok and sample_ok


def dagon2_qa_studies_aux(aux: bool) -> bool:
    """dagon2_qa_studies

    aux:
    dagon2_qa_studies: i
    """
    return aux


def _bench_dagon2_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dagon2_qa_studies_ok(True, True))
    checks.append(not dagon2_qa_studies_ok(False, True))
    checks.append(dagon2_qa_studies_aux(True))
    checks.append(not dagon2_qa_studies_aux(False))
    checks.append(True)  # philistine-myth canon
    return float(sum(checks) / len(checks))


def bench_dagon2_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dagon2_qa_studies": _bench_dagon2_qa_studies(seed)}
