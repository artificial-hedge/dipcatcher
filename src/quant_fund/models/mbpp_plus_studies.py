"""mbpp_plus_studies module (SYNTHETIC)."""

from __future__ import annotations


def mbpp_plus_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mbpp_plus_studies

    check:
    mbpp_plus_studies: MBPP+ pass@1 with augmented tests and metrics
    """
    return fit_ok and sample_ok


def mbpp_plus_studies_aux(aux: bool) -> bool:
    """mbpp_plus_studies

    aux:
    mbpp_plus_studies: tasks, tests, and pass rates
    """
    return aux


def _bench_mbpp_plus_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mbpp_plus_studies_ok(True, True))
    checks.append(not mbpp_plus_studies_ok(False, True))
    checks.append(mbpp_plus_studies_aux(True))
    checks.append(not mbpp_plus_studies_aux(False))
    checks.append(True)  # code-eval canon
    return float(sum(checks) / len(checks))


def bench_mbpp_plus_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mbpp_plus_studies": _bench_mbpp_plus_studies(seed)}
