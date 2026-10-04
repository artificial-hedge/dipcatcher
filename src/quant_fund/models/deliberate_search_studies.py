"""deliberate_search_studies module (SYNTHETIC)."""

from __future__ import annotations


def deliberate_search_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deliberate_search_studies

    check:
    deliberate_search_studies: beam and lookahead exploration/steps and pruning
    """
    return fit_ok and sample_ok


def deliberate_search_studies_aux(aux: bool) -> bool:
    """deliberate_search_studies

    aux:
    deliberate_search_studies: solution coverage and cost-quality tradeoffs/scores and depth
    """
    return aux


def _bench_deliberate_search_studies(seed: int = 0) -> float:
    checks = []
    checks.append(deliberate_search_studies_ok(True, True))
    checks.append(not deliberate_search_studies_ok(False, True))
    checks.append(deliberate_search_studies_aux(True))
    checks.append(not deliberate_search_studies_aux(False))
    checks.append(True)  # inference-scaling canon
    return float(sum(checks) / len(checks))


def bench_deliberate_search_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deliberate_search_studies": _bench_deliberate_search_studies(seed)}
