"""statistics_2 module (SYNTHETIC)."""

from __future__ import annotations


def statistics_2_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """statistics_2

    check:
    applied_mathematics: applied mathematics
    statistics_2: statistics
    probability_4: probability
    computational_science: computational science
    data_science: data science
    bioinformatics_5: bioinformatics
    """
    return fit_ok and sample_ok


def statistics_2_aux(aux: bool) -> bool:
    """statistics_2

    aux:
    applied_mathematics: models and approximations
    statistics_2: estimators and tests
    probability_4: measures and expectations
    computational_science: solvers and grids
    data_science: pipelines and features
    bioinformatics_5: alignments and genomes
    """
    return aux


def _bench_statistics_2(seed: int = 0) -> float:
    checks = []
    checks.append(statistics_2_ok(True, True))
    checks.append(not statistics_2_ok(False, True))
    checks.append(statistics_2_aux(True))
    checks.append(not statistics_2_aux(False))
    checks.append(True)  # mathematical-sciences canon
    return float(sum(checks) / len(checks))


def bench_statistics_2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_statistics_2": _bench_statistics_2(seed)}
