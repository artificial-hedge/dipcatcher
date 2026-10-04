"""salt_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def salt_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """salt_qa_studies

    check:
    salt_qa_studies: SaltQA metrics
    """
    return fit_ok and sample_ok


def salt_qa_studies_aux(aux: bool) -> bool:
    """salt_qa_studies

    aux:
    salt_qa_studies: salt flats, arid pans, answers, and scores
    """
    return aux


def _bench_salt_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(salt_qa_studies_ok(True, True))
    checks.append(not salt_qa_studies_ok(False, True))
    checks.append(salt_qa_studies_aux(True))
    checks.append(not salt_qa_studies_aux(False))
    checks.append(True)  # camelid-steppe canon
    return float(sum(checks) / len(checks))


def bench_salt_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_salt_qa_studies": _bench_salt_qa_studies(seed)}
