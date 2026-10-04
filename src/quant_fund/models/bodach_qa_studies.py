"""bodach_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def bodach_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """bodach_qa_studies

    check:
    bodach_qa_studies: B
    """
    return fit_ok and sample_ok


def bodach_qa_studies_aux(aux: bool) -> bool:
    """bodach_qa_studies

    aux:
    bodach_qa_studies: o
    """
    return aux


def _bench_bodach_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(bodach_qa_studies_ok(True, True))
    checks.append(not bodach_qa_studies_ok(False, True))
    checks.append(bodach_qa_studies_aux(True))
    checks.append(not bodach_qa_studies_aux(False))
    checks.append(True)  # celtic-demon-3 canon
    return float(sum(checks) / len(checks))


def bench_bodach_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bodach_qa_studies": _bench_bodach_qa_studies(seed)}
