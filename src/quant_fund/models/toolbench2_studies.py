"""toolbench2_studies module (SYNTHETIC)."""

from __future__ import annotations


def toolbench2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """toolbench2_studies

    check:
    toolbench2_studies: ToolBench metrics
    """
    return fit_ok and sample_ok


def toolbench2_studies_aux(aux: bool) -> bool:
    """toolbench2_studies

    aux:
    toolbench2_studies: instructions, apis, traces, and scores
    """
    return aux


def _bench_toolbench2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(toolbench2_studies_ok(True, True))
    checks.append(not toolbench2_studies_ok(False, True))
    checks.append(toolbench2_studies_aux(True))
    checks.append(not toolbench2_studies_aux(False))
    checks.append(True)  # toolbench canon
    return float(sum(checks) / len(checks))


def bench_toolbench2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toolbench2_studies": _bench_toolbench2_studies(seed)}
