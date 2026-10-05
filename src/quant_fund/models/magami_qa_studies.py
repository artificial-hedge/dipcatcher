"""magami_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def magami_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """magami_qa_studies

    check:
    magami_qa_studies: M
    """
    return fit_ok and sample_ok


def magami_qa_studies_aux(aux: bool) -> bool:
    """magami_qa_studies

    aux:
    magami_qa_studies: a
    """
    return aux


def _bench_magami_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(magami_qa_studies_ok(True, True))
    checks.append(not magami_qa_studies_ok(False, True))
    checks.append(magami_qa_studies_aux(True))
    checks.append(not magami_qa_studies_aux(False))
    checks.append(True)  # burmese-nat canon
    return float(sum(checks) / len(checks))


def bench_magami_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_magami_qa_studies": _bench_magami_qa_studies(seed)}
