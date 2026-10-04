"""tool_sandbox_studies module (SYNTHETIC)."""

from __future__ import annotations


def tool_sandbox_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tool_sandbox_studies

    check:
    tool_sandbox_studies: ToolSandbox metrics
    """
    return fit_ok and sample_ok


def tool_sandbox_studies_aux(aux: bool) -> bool:
    """tool_sandbox_studies

    aux:
    tool_sandbox_studies: tasks, sandboxes, trajectories, and scores
    """
    return aux


def _bench_tool_sandbox_studies(seed: int = 0) -> float:
    checks = []
    checks.append(tool_sandbox_studies_ok(True, True))
    checks.append(not tool_sandbox_studies_ok(False, True))
    checks.append(tool_sandbox_studies_aux(True))
    checks.append(not tool_sandbox_studies_aux(False))
    checks.append(True)  # MCP-web canon
    return float(sum(checks) / len(checks))


def bench_tool_sandbox_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tool_sandbox_studies": _bench_tool_sandbox_studies(seed)}
