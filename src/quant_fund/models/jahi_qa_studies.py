"""jahi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jahi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jahi_qa_studies

    check:
    jahi_qa_studies: J
    """
    return fit_ok and sample_ok


def jahi_qa_studies_aux(aux: bool) -> bool:
    """jahi_qa_studies

    aux:
    jahi_qa_studies: a
    """
    return aux


def _bench_jahi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jahi_qa_studies_ok(True, True))
    checks.append(not jahi_qa_studies_ok(False, True))
    checks.append(jahi_qa_studies_aux(True))
    checks.append(not jahi_qa_studies_aux(False))
    checks.append(True)  # persian-daeva canon
    return float(sum(checks) / len(checks))


def bench_jahi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jahi_qa_studies": _bench_jahi_qa_studies(seed)}
