"""visual_web_studies module (SYNTHETIC)."""

from __future__ import annotations


def visual_web_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """visual_web_studies

    check:
    visual_web_studies: Visual web-agent metrics
    """
    return fit_ok and sample_ok


def visual_web_studies_aux(aux: bool) -> bool:
    """visual_web_studies

    aux:
    visual_web_studies: screenshots, layouts, clicks, and hit rates
    """
    return aux


def _bench_visual_web_studies(seed: int = 0) -> float:
    checks = []
    checks.append(visual_web_studies_ok(True, True))
    checks.append(not visual_web_studies_ok(False, True))
    checks.append(visual_web_studies_aux(True))
    checks.append(not visual_web_studies_aux(False))
    checks.append(True)  # agentic-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_visual_web_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_visual_web_studies": _bench_visual_web_studies(seed)}
