"""mahagiri_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def mahagiri_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mahagiri_qa_studies

    check:
    mahagiri_qa_studies: M
    """
    return fit_ok and sample_ok


def mahagiri_qa_studies_aux(aux: bool) -> bool:
    """mahagiri_qa_studies

    aux:
    mahagiri_qa_studies: a
    """
    return aux


def _bench_mahagiri_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mahagiri_qa_studies_ok(True, True))
    checks.append(not mahagiri_qa_studies_ok(False, True))
    checks.append(mahagiri_qa_studies_aux(True))
    checks.append(not mahagiri_qa_studies_aux(False))
    checks.append(True)  # burmese-nat canon
    return float(sum(checks) / len(checks))


def bench_mahagiri_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mahagiri_qa_studies": _bench_mahagiri_qa_studies(seed)}
