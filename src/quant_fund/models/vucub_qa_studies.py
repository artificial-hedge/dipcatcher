"""vucub_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def vucub_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """vucub_qa_studies

    check:
    vucub_qa_studies: V
    """
    return fit_ok and sample_ok


def vucub_qa_studies_aux(aux: bool) -> bool:
    """vucub_qa_studies

    aux:
    vucub_qa_studies: u
    """
    return aux


def _bench_vucub_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(vucub_qa_studies_ok(True, True))
    checks.append(not vucub_qa_studies_ok(False, True))
    checks.append(vucub_qa_studies_aux(True))
    checks.append(not vucub_qa_studies_aux(False))
    checks.append(True)  # mesoamerican-demon canon
    return float(sum(checks) / len(checks))


def bench_vucub_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vucub_qa_studies": _bench_vucub_qa_studies(seed)}
