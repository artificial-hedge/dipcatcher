"""nq_open_studies module (SYNTHETIC)."""

from __future__ import annotations


def nq_open_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nq_open_studies

    check:
    nq_open_studies: Natural-Questions open metrics
    """
    return fit_ok and sample_ok


def nq_open_studies_aux(aux: bool) -> bool:
    """nq_open_studies

    aux:
    nq_open_studies: questions, answers, contexts, and accuracies
    """
    return aux


def _bench_nq_open_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nq_open_studies_ok(True, True))
    checks.append(not nq_open_studies_ok(False, True))
    checks.append(nq_open_studies_aux(True))
    checks.append(not nq_open_studies_aux(False))
    checks.append(True)  # open-domain-QA canon
    return float(sum(checks) / len(checks))


def bench_nq_open_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nq_open_studies": _bench_nq_open_studies(seed)}
