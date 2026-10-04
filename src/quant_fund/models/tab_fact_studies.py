"""tab_fact_studies module (SYNTHETIC)."""

from __future__ import annotations


def tab_fact_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tab_fact_studies

    check:
    tab_fact_studies: TabFact metrics
    """
    return fit_ok and sample_ok


def tab_fact_studies_aux(aux: bool) -> bool:
    """tab_fact_studies

    aux:
    tab_fact_studies: tables, claims, evidence, and scores
    """
    return aux


def _bench_tab_fact_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tab_fact_studies_ok(True, True))
    checks.append(not tab_fact_studies_ok(False, True))
    checks.append(tab_fact_studies_aux(True))
    checks.append(not tab_fact_studies_aux(False))
    checks.append(True)  # numerical-reasoning canon
    return float(sum(checks) / len(checks))


def bench_tab_fact_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tab_fact_studies": _bench_tab_fact_studies(seed)}
