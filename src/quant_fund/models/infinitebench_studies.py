"""infinitebench_studies module (SYNTHETIC)."""

from __future__ import annotations


def infinitebench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """infinitebench_studies

    check:
    infinitebench_studies: InfiniteBench 100K+-token retrieval and reasoning scores
    """
    return fit_ok and sample_ok


def infinitebench_studies_aux(aux: bool) -> bool:
    """infinitebench_studies

    aux:
    infinitebench_studies: documents, queries, and accuracy metrics
    """
    return aux


def _bench_infinitebench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(infinitebench_studies_ok(True, True))
    checks.append(not infinitebench_studies_ok(False, True))
    checks.append(infinitebench_studies_aux(True))
    checks.append(not infinitebench_studies_aux(False))
    checks.append(True)  # long-context-eval canon
    return float(sum(checks) / len(checks))


def bench_infinitebench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_infinitebench_studies": _bench_infinitebench_studies(seed)}
