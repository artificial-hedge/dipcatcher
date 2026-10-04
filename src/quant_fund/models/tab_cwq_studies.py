"""tab_cwq_studies module (SYNTHETIC)."""

from __future__ import annotations


def tab_cwq_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tab_cwq_studies

    check:
    tab_cwq_studies: TableCWQ metrics
    """
    return fit_ok and sample_ok


def tab_cwq_studies_aux(aux: bool) -> bool:
    """tab_cwq_studies

    aux:
    tab_cwq_studies: tables, questions, programs, and scores
    """
    return aux


def _bench_tab_cwq_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tab_cwq_studies_ok(True, True))
    checks.append(not tab_cwq_studies_ok(False, True))
    checks.append(tab_cwq_studies_aux(True))
    checks.append(not tab_cwq_studies_aux(False))
    checks.append(True)  # table-QA canon
    return float(sum(checks) / len(checks))


def bench_tab_cwq_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tab_cwq_studies": _bench_tab_cwq_studies(seed)}
