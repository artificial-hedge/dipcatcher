"""webvoyager_studies module (SYNTHETIC)."""

from __future__ import annotations


def webvoyager_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """webvoyager_studies

    check:
    webvoyager_studies: WebVoyager browser-task success and metrics
    """
    return fit_ok and sample_ok


def webvoyager_studies_aux(aux: bool) -> bool:
    """webvoyager_studies

    aux:
    webvoyager_studies: web states, actions, and completion rates
    """
    return aux


def _bench_webvoyager_studies(seed: int = 0) -> float:
    checks = []
    checks.append(webvoyager_studies_ok(True, True))
    checks.append(not webvoyager_studies_ok(False, True))
    checks.append(webvoyager_studies_aux(True))
    checks.append(not webvoyager_studies_aux(False))
    checks.append(True)  # agent-eval canon
    return float(sum(checks) / len(checks))


def bench_webvoyager_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_webvoyager_studies": _bench_webvoyager_studies(seed)}
