"""web_nav_studies module (SYNTHETIC)."""

from __future__ import annotations


def web_nav_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """web_nav_studies

    check:
    web_nav_studies: Web navigation agent metrics
    """
    return fit_ok and sample_ok


def web_nav_studies_aux(aux: bool) -> bool:
    """web_nav_studies

    aux:
    web_nav_studies: pages, actions, observations, and success rates
    """
    return aux


def _bench_web_nav_studies(seed: int = 0) -> float:
    checks = []
    checks.append(web_nav_studies_ok(True, True))
    checks.append(not web_nav_studies_ok(False, True))
    checks.append(web_nav_studies_aux(True))
    checks.append(not web_nav_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_web_nav_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_web_nav_studies": _bench_web_nav_studies(seed)}
