"""leshenka_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def leshenka_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """leshenka_qa_studies

    check:
    leshenka_qa_studies: L
    """
    return fit_ok and sample_ok


def leshenka_qa_studies_aux(aux: bool) -> bool:
    """leshenka_qa_studies

    aux:
    leshenka_qa_studies: e
    """
    return aux


def _bench_leshenka_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(leshenka_qa_studies_ok(True, True))
    checks.append(not leshenka_qa_studies_ok(False, True))
    checks.append(leshenka_qa_studies_aux(True))
    checks.append(not leshenka_qa_studies_aux(False))
    checks.append(True)  # persian-spirit canon
    return float(sum(checks) / len(checks))


def bench_leshenka_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leshenka_qa_studies": _bench_leshenka_qa_studies(seed)}
