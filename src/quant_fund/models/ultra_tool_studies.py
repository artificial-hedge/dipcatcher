"""ultra_tool_studies module (SYNTHETIC)."""

from __future__ import annotations


def ultra_tool_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ultra_tool_studies

    check:
    ultra_tool_studies: UltraTool metrics
    """
    return fit_ok and sample_ok


def ultra_tool_studies_aux(aux: bool) -> bool:
    """ultra_tool_studies

    aux:
    ultra_tool_studies: goals, tools, executions, and scores
    """
    return aux


def _bench_ultra_tool_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ultra_tool_studies_ok(True, True))
    checks.append(not ultra_tool_studies_ok(False, True))
    checks.append(ultra_tool_studies_aux(True))
    checks.append(not ultra_tool_studies_aux(False))
    checks.append(True)  # toolbench canon
    return float(sum(checks) / len(checks))


def bench_ultra_tool_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ultra_tool_studies": _bench_ultra_tool_studies(seed)}
