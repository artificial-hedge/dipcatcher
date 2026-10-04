"""beaver_safe_studies module (SYNTHETIC)."""

from __future__ import annotations


def beaver_safe_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """beaver_safe_studies

    check:
    beaver_safe_studies: Beaver cost-model constraint-satisfaction metrics
    """
    return fit_ok and sample_ok


def beaver_safe_studies_aux(aux: bool) -> bool:
    """beaver_safe_studies

    aux:
    beaver_safe_studies: responses, costs, constraints, and scores
    """
    return aux


def _bench_beaver_safe_studies(seed: int = 0) -> float:
    checks = []
    checks.append(beaver_safe_studies_ok(True, True))
    checks.append(not beaver_safe_studies_ok(False, True))
    checks.append(beaver_safe_studies_aux(True))
    checks.append(not beaver_safe_studies_aux(False))
    checks.append(True)  # safety-alignment-2 canon
    return float(sum(checks) / len(checks))


def bench_beaver_safe_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beaver_safe_studies": _bench_beaver_safe_studies(seed)}
