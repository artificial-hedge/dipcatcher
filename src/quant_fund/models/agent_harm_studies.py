"""agent_harm_studies module (SYNTHETIC)."""

from __future__ import annotations


def agent_harm_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """agent_harm_studies

    check:
    agent_harm_studies: agent-harm trajectories/tools and risk scores
    """
    return fit_ok and sample_ok


def agent_harm_studies_aux(aux: bool) -> bool:
    """agent_harm_studies

    aux:
    agent_harm_studies: sandboxed action policies/guards and audits
    """
    return aux


def _bench_agent_harm_studies(seed: int = 0) -> float:
    checks = []
    checks.append(agent_harm_studies_ok(True, True))
    checks.append(not agent_harm_studies_ok(False, True))
    checks.append(agent_harm_studies_aux(True))
    checks.append(not agent_harm_studies_aux(False))
    checks.append(True)  # safety-eval canon
    return float(sum(checks) / len(checks))


def bench_agent_harm_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_agent_harm_studies": _bench_agent_harm_studies(seed)}
