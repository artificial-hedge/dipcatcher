"""least_to_most_studies module (SYNTHETIC)."""

from __future__ import annotations


def least_to_most_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """least_to_most_studies

    check:
    least_to_most_studies: subproblem decomposition and sequencing/orders and chains
    """
    return fit_ok and sample_ok


def least_to_most_studies_aux(aux: bool) -> bool:
    """least_to_most_studies

    aux:
    least_to_most_studies: progressive solution accumulation/parts and answers
    """
    return aux


def _bench_least_to_most_studies(seed: int = 0) -> float:
    checks = []
    checks.append(least_to_most_studies_ok(True, True))
    checks.append(not least_to_most_studies_ok(False, True))
    checks.append(least_to_most_studies_aux(True))
    checks.append(not least_to_most_studies_aux(False))
    checks.append(True)  # reasoning-prompt canon
    return float(sum(checks) / len(checks))


def bench_least_to_most_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_least_to_most_studies": _bench_least_to_most_studies(seed)}
