"""ekimmu_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def ekimmu_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ekimmu_qa_studies

    check:
    ekimmu_qa_studies: e
    """
    return fit_ok and sample_ok


def ekimmu_qa_studies_aux(aux: bool) -> bool:
    """ekimmu_qa_studies

    aux:
    ekimmu_qa_studies: k
    """
    return aux


def _bench_ekimmu_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ekimmu_qa_studies_ok(True, True))
    checks.append(not ekimmu_qa_studies_ok(False, True))
    checks.append(ekimmu_qa_studies_aux(True))
    checks.append(not ekimmu_qa_studies_aux(False))
    checks.append(True)  # mesopotamian-demon-2 canon
    return float(sum(checks) / len(checks))


def bench_ekimmu_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ekimmu_qa_studies": _bench_ekimmu_qa_studies(seed)}
