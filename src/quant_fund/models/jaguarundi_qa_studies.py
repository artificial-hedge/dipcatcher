"""jaguarundi_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def jaguarundi_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """jaguarundi_qa_studies

    check:
    jaguarundi_qa_studies: JaguarundiQA metrics
    """
    return fit_ok and sample_ok


def jaguarundi_qa_studies_aux(aux: bool) -> bool:
    """jaguarundi_qa_studies

    aux:
    jaguarundi_qa_studies: jaguarundis, pelts, answers, and scores
    """
    return aux


def _bench_jaguarundi_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(jaguarundi_qa_studies_ok(True, True))
    checks.append(not jaguarundi_qa_studies_ok(False, True))
    checks.append(jaguarundi_qa_studies_aux(True))
    checks.append(not jaguarundi_qa_studies_aux(False))
    checks.append(True)  # wildcat canon
    return float(sum(checks) / len(checks))


def bench_jaguarundi_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jaguarundi_qa_studies": _bench_jaguarundi_qa_studies(seed)}
