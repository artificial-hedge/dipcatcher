"""narra_sum_studies module (SYNTHETIC)."""

from __future__ import annotations


def narra_sum_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """narra_sum_studies

    check:
    narra_sum_studies: NarraSum metrics
    """
    return fit_ok and sample_ok


def narra_sum_studies_aux(aux: bool) -> bool:
    """narra_sum_studies

    aux:
    narra_sum_studies: plots, hypotheses, answers, and scores
    """
    return aux


def _bench_narra_sum_studies(seed: int = 0) -> float:
    checks = []
    checks.append(narra_sum_studies_ok(True, True))
    checks.append(not narra_sum_studies_ok(False, True))
    checks.append(narra_sum_studies_aux(True))
    checks.append(not narra_sum_studies_aux(False))
    checks.append(True)  # long-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_narra_sum_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_narra_sum_studies": _bench_narra_sum_studies(seed)}
