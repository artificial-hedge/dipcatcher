"""gorilla_qa_studies module (SYNTHETIC)."""

from __future__ import annotations


def gorilla_qa_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """gorilla_qa_studies

    check:
    gorilla_qa_studies: GorillaQA metrics
    """
    return fit_ok and sample_ok


def gorilla_qa_studies_aux(aux: bool) -> bool:
    """gorilla_qa_studies

    aux:
    gorilla_qa_studies: gorillas, troops, answers, and scores
    """
    return aux


def _bench_gorilla_qa_studies(seed: int = 0) -> float:
    checks = []
    checks.append(gorilla_qa_studies_ok(True, True))
    checks.append(not gorilla_qa_studies_ok(False, True))
    checks.append(gorilla_qa_studies_aux(True))
    checks.append(not gorilla_qa_studies_aux(False))
    checks.append(True)  # jungle canon
    return float(sum(checks) / len(checks))


def bench_gorilla_qa_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gorilla_qa_studies": _bench_gorilla_qa_studies(seed)}
