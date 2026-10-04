"""doc2dial_studies module (SYNTHETIC)."""

from __future__ import annotations


def doc2dial_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """doc2dial_studies

    check:
    doc2dial_studies: Doc2Dial metrics
    """
    return fit_ok and sample_ok


def doc2dial_studies_aux(aux: bool) -> bool:
    """doc2dial_studies

    aux:
    doc2dial_studies: docs, turns, answers, and scores
    """
    return aux


def _bench_doc2dial_studies(seed: int = 0) -> float:
    checks = []
    checks.append(doc2dial_studies_ok(True, True))
    checks.append(not doc2dial_studies_ok(False, True))
    checks.append(doc2dial_studies_aux(True))
    checks.append(not doc2dial_studies_aux(False))
    checks.append(True)  # table-QA canon
    return float(sum(checks) / len(checks))


def bench_doc2dial_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doc2dial_studies": _bench_doc2dial_studies(seed)}
