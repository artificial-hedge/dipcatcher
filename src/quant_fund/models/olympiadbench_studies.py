"""olympiadbench_studies module (SYNTHETIC)."""

from __future__ import annotations


def olympiadbench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """olympiadbench_studies

    check:
    olympiadbench_studies: OlympiadBench metrics
    """
    return fit_ok and sample_ok


def olympiadbench_studies_aux(aux: bool) -> bool:
    """olympiadbench_studies

    aux:
    olympiadbench_studies: problems, rubrics, solutions, and scores
    """
    return aux


def _bench_olympiadbench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(olympiadbench_studies_ok(True, True))
    checks.append(not olympiadbench_studies_ok(False, True))
    checks.append(olympiadbench_studies_aux(True))
    checks.append(not olympiadbench_studies_aux(False))
    checks.append(True)  # frontier-eval canon
    return float(sum(checks) / len(checks))


def bench_olympiadbench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_olympiadbench_studies": _bench_olympiadbench_studies(seed)}
