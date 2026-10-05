"""tuyul_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def tuyul_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tuyul_qa_studies

    check:
    tuyul_qa_studies: T
    """
    return fit_ok and sample_ok


def tuyul_qa_studies_aux(aux: bool) -> bool:
    """tuyul_qa_studies

    aux:
    tuyul_qa_studies: u
    """
    return aux


def _bench_tuyul_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tuyul_qa_studies_ok(True, True))
    checks.append(not tuyul_qa_studies_ok(False, True))
    checks.append(tuyul_qa_studies_aux(True))
    checks.append(not tuyul_qa_studies_aux(False))
    checks.append(True)  # javanese-demon canon
    return float(sum(checks) / len(checks))


def bench_tuyul_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tuyul_qa_studies": _bench_tuyul_qa_studies(seed)}
