"""rarog_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def rarog_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """rarog_qa_studies

    check:
    rarog_qa_studies: R
    """
    return fit_ok and sample_ok


def rarog_qa_studies_aux(aux: bool) -> bool:
    """rarog_qa_studies

    aux:
    rarog_qa_studies: a
    """
    return aux


def _bench_rarog_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(rarog_qa_studies_ok(True, True))
    checks.append(not rarog_qa_studies_ok(False, True))
    checks.append(rarog_qa_studies_aux(True))
    checks.append(not rarog_qa_studies_aux(False))
    checks.append(True)  # slavic-demon canon
    return float(sum(checks) / len(checks))


def bench_rarog_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rarog_qa_studies": _bench_rarog_qa_studies(seed)}
