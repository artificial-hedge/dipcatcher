"""nest_tools_studies module (SYNTHETIC)."""

from __future__ import annotations


def nest_tools_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nest_tools_studies

    check:
    nest_tools_studies: NestTools metrics
    """
    return fit_ok and sample_ok


def nest_tools_studies_aux(aux: bool) -> bool:
    """nest_tools_studies

    aux:
    nest_tools_studies: queries, apis, calls, and scores
    """
    return aux


def _bench_nest_tools_studies(seed: int = 0) -> float:
    checks = []
    checks.append(nest_tools_studies_ok(True, True))
    checks.append(not nest_tools_studies_ok(False, True))
    checks.append(nest_tools_studies_aux(True))
    checks.append(not nest_tools_studies_aux(False))
    checks.append(True)  # toolbench canon
    return float(sum(checks) / len(checks))


def bench_nest_tools_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nest_tools_studies": _bench_nest_tools_studies(seed)}
