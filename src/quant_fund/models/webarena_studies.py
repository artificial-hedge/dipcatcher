"""webarena_studies module (SYNTHETIC)."""

from __future__ import annotations


def webarena_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """webarena_studies

    check:
    webarena_studies: WebArena environment metrics
    """
    return fit_ok and sample_ok


def webarena_studies_aux(aux: bool) -> bool:
    """webarena_studies

    aux:
    webarena_studies: sites, intents, states, and scores
    """
    return aux


def _bench_webarena_studies(seed: int = 0) -> float:
    checks = []
    checks.append(webarena_studies_ok(True, True))
    checks.append(not webarena_studies_ok(False, True))
    checks.append(webarena_studies_aux(True))
    checks.append(not webarena_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_webarena_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_webarena_studies": _bench_webarena_studies(seed)}
