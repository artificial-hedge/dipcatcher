"""web_agent_studies module (SYNTHETIC)."""

from __future__ import annotations


def web_agent_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """web_agent_studies

    check:
    web_agent_studies: DOM navigation and form filling/browsing and extraction
    """
    return fit_ok and sample_ok


def web_agent_studies_aux(aux: bool) -> bool:
    """web_agent_studies

    aux:
    web_agent_studies: task success and grounding/elements and workflows
    """
    return aux


def _bench_web_agent_studies(seed: int = 0) -> float:
    checks = []
    checks.append(web_agent_studies_ok(True, True))
    checks.append(not web_agent_studies_ok(False, True))
    checks.append(web_agent_studies_aux(True))
    checks.append(not web_agent_studies_aux(False))
    checks.append(True)  # agent-infrastructure canon
    return float(sum(checks) / len(checks))


def bench_web_agent_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_web_agent_studies": _bench_web_agent_studies(seed)}
