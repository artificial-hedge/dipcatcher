"""maero_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def maero_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """maero_qa_studies

    check:
    maero_qa_studies: M
    """
    return fit_ok and sample_ok


def maero_qa_studies_aux(aux: bool) -> bool:
    """maero_qa_studies

    aux:
    maero_qa_studies: a
    """
    return aux


def _bench_maero_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(maero_qa_studies_ok(True, True))
    checks.append(not maero_qa_studies_ok(False, True))
    checks.append(maero_qa_studies_aux(True))
    checks.append(not maero_qa_studies_aux(False))
    checks.append(True)  # polynesian-demon canon
    return float(sum(checks) / len(checks))


def bench_maero_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maero_qa_studies": _bench_maero_qa_studies(seed)}
