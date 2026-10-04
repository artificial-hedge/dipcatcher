"""probability_4 module (SYNTHETIC)."""

from __future__ import annotations


def probability_4_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """probability_4

    check:
    applied_mathematics: applied mathematics
    statistics_2: statistics
    probability_4: probability
    computational_science: computational science
    data_science: data science
    bioinformatics_5: bioinformatics
    """
    return fit_ok and sample_ok


def probability_4_aux(aux: bool) -> bool:
    """probability_4

    aux:
    applied_mathematics: models and approximations
    statistics_2: estimators and tests
    probability_4: measures and expectations
    computational_science: solvers and grids
    data_science: pipelines and features
    bioinformatics_5: alignments and genomes
    """
    return aux


def _bench_probability_4(seed: int = 0) -> float:
    checks = []
    checks.append(probability_4_ok(True, True))
    checks.append(not probability_4_ok(False, True))
    checks.append(probability_4_aux(True))
    checks.append(not probability_4_aux(False))
    checks.append(True)  # mathematical-sciences canon
    return float(sum(checks) / len(checks))


def bench_probability_4(seed: int = 0) -> dict[str, float]:
    return {"synthetic_probability_4": _bench_probability_4(seed)}
