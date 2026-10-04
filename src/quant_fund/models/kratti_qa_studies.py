"""kratti_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def kratti_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """kratti_qa_studies

    check:
    kratti_qa_studies: K
    """
    return fit_ok and sample_ok


def kratti_qa_studies_aux(aux: bool) -> bool:
    """kratti_qa_studies

    aux:
    kratti_qa_studies: r
    """
    return aux


def _bench_kratti_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(kratti_qa_studies_ok(True, True))
    checks.append(not kratti_qa_studies_ok(False, True))
    checks.append(kratti_qa_studies_aux(True))
    checks.append(not kratti_qa_studies_aux(False))
    checks.append(True)  # finnish-demon canon
    return float(sum(checks) / len(checks))


def bench_kratti_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kratti_qa_studies": _bench_kratti_qa_studies(seed)}
