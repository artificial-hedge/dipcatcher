"""data_science module (SYNTHETIC)."""

from __future__ import annotations


def data_science_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """data_science

    check:
    applied_mathematics: applied mathematics
    statistics_2: statistics
    probability_4: probability
    computational_science: computational science
    data_science: data science
    bioinformatics_5: bioinformatics
    """
    return fit_ok and sample_ok


def data_science_aux(aux: bool) -> bool:
    """data_science

    aux:
    applied_mathematics: models and approximations
    statistics_2: estimators and tests
    probability_4: measures and expectations
    computational_science: solvers and grids
    data_science: pipelines and features
    bioinformatics_5: alignments and genomes
    """
    return aux


def _bench_data_science(seed: int = 0) -> float:
    checks = []
    checks.append(data_science_ok(True, True))
    checks.append(not data_science_ok(False, True))
    checks.append(data_science_aux(True))
    checks.append(not data_science_aux(False))
    checks.append(True)  # mathematical-sciences canon
    return float(sum(checks) / len(checks))


def bench_data_science(seed: int = 0) -> dict[str, float]:
    return {"synthetic_data_science": _bench_data_science(seed)}
