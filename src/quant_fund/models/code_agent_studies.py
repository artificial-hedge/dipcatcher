"""code_agent_studies module (SYNTHETIC)."""

from __future__ import annotations


def code_agent_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """code_agent_studies

    check:
    code_agent_studies: repo editing and patch generation/tests and review
    """
    return fit_ok and sample_ok


def code_agent_studies_aux(aux: bool) -> bool:
    """code_agent_studies

    aux:
    code_agent_studies: swe-bench-style loops and verification/diffs and iterates
    """
    return aux


def _bench_code_agent_studies(seed: int = 0) -> float:
    checks = []
    checks.append(code_agent_studies_ok(True, True))
    checks.append(not code_agent_studies_ok(False, True))
    checks.append(code_agent_studies_aux(True))
    checks.append(not code_agent_studies_aux(False))
    checks.append(True)  # agent-infrastructure canon
    return float(sum(checks) / len(checks))


def bench_code_agent_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_code_agent_studies": _bench_code_agent_studies(seed)}
