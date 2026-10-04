"""best_of_n_studies module (SYNTHETIC)."""

from __future__ import annotations


def best_of_n_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """best_of_n_studies

    check:
    best_of_n_studies: rejection sampling and reward ranking/sampling and selection
    """
    return fit_ok and sample_ok


def best_of_n_studies_aux(aux: bool) -> bool:
    """best_of_n_studies

    aux:
    best_of_n_studies: distillation and curriculum/diversity and coverage
    """
    return aux


def _bench_best_of_n_studies(seed: int = 0) -> float:
    checks = []
    checks.append(best_of_n_studies_ok(True, True))
    checks.append(not best_of_n_studies_ok(False, True))
    checks.append(best_of_n_studies_aux(True))
    checks.append(not best_of_n_studies_aux(False))
    checks.append(True)  # post-training-2 canon
    return float(sum(checks) / len(checks))


def bench_best_of_n_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_best_of_n_studies": _bench_best_of_n_studies(seed)}
