"""frontier_math_studies module (SYNTHETIC)."""

from __future__ import annotations


def frontier_math_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """frontier_math_studies

    check:
    frontier_math_studies: FrontierMath research problems, tiers, and solve rate
    """
    return fit_ok and sample_ok


def frontier_math_studies_aux(aux: bool) -> bool:
    """frontier_math_studies

    aux:
    frontier_math_studies: T1-T4 problems, verifiers, and partial credit
    """
    return aux


def _bench_frontier_math_studies(seed: int = 0) -> float:
    checks = []
    checks.append(frontier_math_studies_ok(True, True))
    checks.append(not frontier_math_studies_ok(False, True))
    checks.append(frontier_math_studies_aux(True))
    checks.append(not frontier_math_studies_aux(False))
    checks.append(True)  # hard-benchmark canon
    return float(sum(checks) / len(checks))


def bench_frontier_math_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frontier_math_studies": _bench_frontier_math_studies(seed)}
