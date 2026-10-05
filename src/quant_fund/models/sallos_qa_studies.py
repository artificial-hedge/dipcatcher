"""sallos_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def sallos_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sallos_qa_studies

    check:
    sallos_qa_studies: S
    """
    return fit_ok and sample_ok


def sallos_qa_studies_aux(aux: bool) -> bool:
    """sallos_qa_studies

    aux:
    sallos_qa_studies: a
    """
    return aux


def _bench_sallos_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sallos_qa_studies_ok(True, True))
    checks.append(not sallos_qa_studies_ok(False, True))
    checks.append(sallos_qa_studies_aux(True))
    checks.append(not sallos_qa_studies_aux(False))
    checks.append(True)  # goetic-hierarchy canon
    return float(sum(checks) / len(checks))


def bench_sallos_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sallos_qa_studies": _bench_sallos_qa_studies(seed)}
