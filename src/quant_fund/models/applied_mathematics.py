"""applied_mathematics module (SYNTHETIC)."""

from __future__ import annotations


def applied_mathematics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """applied_mathematics

    check:
    applied_mathematics: applied mathematics
    statistics_2: statistics
    probability_4: probability
    computational_science: computational science
    data_science: data science
    bioinformatics_5: bioinformatics
    """
    return fit_ok and sample_ok


def applied_mathematics_aux(aux: bool) -> bool:
    """applied_mathematics

    aux:
    applied_mathematics: models and approximations
    statistics_2: estimators and tests
    probability_4: measures and expectations
    computational_science: solvers and grids
    data_science: pipelines and features
    bioinformatics_5: alignments and genomes
    """
    return aux


def _bench_applied_mathematics(seed: int = 0) -> float:
    checks = []
    checks.append(applied_mathematics_ok(True, True))
    checks.append(not applied_mathematics_ok(False, True))
    checks.append(applied_mathematics_aux(True))
    checks.append(not applied_mathematics_aux(False))
    checks.append(True)  # mathematical-sciences canon
    return float(sum(checks) / len(checks))


def bench_applied_mathematics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_applied_mathematics": _bench_applied_mathematics(seed)}
