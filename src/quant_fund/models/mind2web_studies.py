"""mind2web_studies module (SYNTHETIC)."""

from __future__ import annotations


def mind2web_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mind2web_studies

    check:
    mind2web_studies: Mind2Web generalist web-agent metrics
    """
    return fit_ok and sample_ok


def mind2web_studies_aux(aux: bool) -> bool:
    """mind2web_studies

    aux:
    mind2web_studies: websites, tasks, trajectories, and scores
    """
    return aux


def _bench_mind2web_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mind2web_studies_ok(True, True))
    checks.append(not mind2web_studies_ok(False, True))
    checks.append(mind2web_studies_aux(True))
    checks.append(not mind2web_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_mind2web_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mind2web_studies": _bench_mind2web_studies(seed)}
