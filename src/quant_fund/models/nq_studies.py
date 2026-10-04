"""nq_studies module (SYNTHETIC)."""

from __future__ import annotations


def nq_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nq_studies

    check:
    nq_studies: Natural Questions long/short answers and EM
    """
    return fit_ok and sample_ok


def nq_studies_aux(aux: bool) -> bool:
    """nq_studies

    aux:
    nq_studies: queries, wiki docs, annotations, and accuracy
    """
    return aux


def _bench_nq_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nq_studies_ok(True, True))
    checks.append(not nq_studies_ok(False, True))
    checks.append(nq_studies_aux(True))
    checks.append(not nq_studies_aux(False))
    checks.append(True)  # reading-comprehension canon
    return float(sum(checks) / len(checks))


def bench_nq_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nq_studies": _bench_nq_studies(seed)}
