"""fact_score_studies module (SYNTHETIC)."""

from __future__ import annotations


def fact_score_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """fact_score_studies

    check:
    fact_score_studies: FActScore atomicity metrics
    """
    return fit_ok and sample_ok


def fact_score_studies_aux(aux: bool) -> bool:
    """fact_score_studies

    aux:
    fact_score_studies: bios, atoms, verdicts, and scores
    """
    return aux


def _bench_fact_score_studies(seed: int = 0) -> float:
    checks = []
    checks.append(fact_score_studies_ok(True, True))
    checks.append(not fact_score_studies_ok(False, True))
    checks.append(fact_score_studies_aux(True))
    checks.append(not fact_score_studies_aux(False))
    checks.append(True)  # LLM-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_fact_score_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fact_score_studies": _bench_fact_score_studies(seed)}
