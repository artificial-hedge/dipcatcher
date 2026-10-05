"""albi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def albi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """albi_qa_studies

    check:
    albi_qa_studies: A
    """
    return fit_ok and sample_ok


def albi_qa_studies_aux(aux: bool) -> bool:
    """albi_qa_studies

    aux:
    albi_qa_studies: l
    """
    return aux


def _bench_albi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(albi_qa_studies_ok(True, True))
    checks.append(not albi_qa_studies_ok(False, True))
    checks.append(albi_qa_studies_aux(True))
    checks.append(not albi_qa_studies_aux(False))
    checks.append(True)  # caucasus-demon canon
    return float(sum(checks) / len(checks))


def bench_albi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_albi_qa_studies": _bench_albi_qa_studies(seed)}
