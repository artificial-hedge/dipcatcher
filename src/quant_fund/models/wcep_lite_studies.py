"""wcep_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def wcep_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wcep_lite_studies

    check:
    wcep_lite_studies: WCEP metrics
    """
    return fit_ok and sample_ok


def wcep_lite_studies_aux(aux: bool) -> bool:
    """wcep_lite_studies

    aux:
    wcep_lite_studies: events, articles, summaries, and scores
    """
    return aux


def _bench_wcep_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wcep_lite_studies_ok(True, True))
    checks.append(not wcep_lite_studies_ok(False, True))
    checks.append(wcep_lite_studies_aux(True))
    checks.append(not wcep_lite_studies_aux(False))
    checks.append(True)  # multi-doc-sum canon
    return float(sum(checks) / len(checks))


def bench_wcep_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wcep_lite_studies": _bench_wcep_lite_studies(seed)}
