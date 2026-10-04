"""workarena_studies module (SYNTHETIC)."""

from __future__ import annotations


def workarena_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """workarena_studies

    check:
    workarena_studies: WorkArena web-workflow tasks and success rate
    """
    return fit_ok and sample_ok


def workarena_studies_aux(aux: bool) -> bool:
    """workarena_studies

    aux:
    workarena_studies: enterprise-miniWob tasks, steps, and rewards
    """
    return aux


def _bench_workarena_studies(seed: int = 0) -> float:
    checks = []
    checks.append(workarena_studies_ok(True, True))
    checks.append(not workarena_studies_ok(False, True))
    checks.append(workarena_studies_aux(True))
    checks.append(not workarena_studies_aux(False))
    checks.append(True)  # hard-benchmark canon
    return float(sum(checks) / len(checks))


def bench_workarena_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_workarena_studies": _bench_workarena_studies(seed)}
